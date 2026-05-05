# watermarks/explain_metrics.py  (replace the previous version)

import os, argparse, pickle, time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, GradientShap, Deconvolution, LRP
from zennit.composites import EpsilonAlpha2Beta1
from scipy.ndimage import sobel, laplace

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

def suffix_for(scale: str, position: str, invert: bool) -> str:
    s = ""
    if scale == "neg_one_one": s += "_rescaled"
    if position == "variable": s += "_variablepos"
    if invert: s += "_inverted"
    return s

def dataset_path(artifacts_dir, split_index, base, subset, scale, position, invert) -> str:
    return os.path.join(artifacts_dir, f"split_{split_index}_{base}_{subset}{suffix_for(scale, position, invert)}.pkl")

def model_path(models_dir, base, split_index, scale, position, invert, seed_value) -> str:
    tag = f"{suffix_for(scale, position, invert)}_split{split_index}".lstrip("_")
    return os.path.join(models_dir, f"cnn_{base}_{tag}_seed{seed_value}.pt")

def get_device(gpu):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{0 if gpu is None else gpu}")
    return torch.device("cpu")

def to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.min() < 0: x = (x + 1.0) / 2.0
    return np.clip(x, 0, 1)

def energy(att: np.ndarray, mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.float32)
    if m.sum() == 0: return 0.0
    img_sz = att.size
    return (float((att*m).sum()) / float(m.sum())) / (float(att.sum()) / float(img_sz) + 1e-12)

def combine_attr_rgb_mean_abs(attr3: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(attr3[..., :3]), axis=-1)

def load_test_sets(artifacts_dir, split_index, scale, position, invert):
    p_all = dataset_path(artifacts_dir, split_index, "all_watermark", "test", scale, position, invert)
    p_none = dataset_path(artifacts_dir, split_index, "no_watermark", "test", scale, position, invert)
    if not os.path.exists(p_all) or not os.path.exists(p_none):
        raise FileNotFoundError(f"Missing test artifacts:\n  {p_all}\n  {p_none}")

    def _load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if len(obj) == 3: data, labels, wm_inds = obj; masks = None
        else: data, labels, wm_inds, masks = obj[:4]
        X = np.transpose(to01(data), (0, 3, 1, 2))
        y = np.asarray(labels).reshape(-1).astype(int)
        return X, y, masks

    X_wm, y_wm, masks_wm = _load(p_all)
    X_no, y_no, _ = _load(p_none)
    return (X_wm, y_wm, masks_wm), (X_no, y_no), p_all, p_none

# --- fixed-position mask derivation from the banner ---
from PIL import Image

def _wm_alpha_fullwidth(wm_path: str, out_width: int, thresh: float) -> np.ndarray:
    wm = Image.open(wm_path).convert("RGB")
    new_h = int(round(wm.size[1] * (out_width / wm.size[0])))
    wm = wm.resize((out_width, new_h))
    rgb = np.asarray(wm, dtype=np.float32)
    rgb = (rgb - rgb.min()) / max(1e-8, (rgb.max() - rgb.min()))
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    alpha = 1.0 - (0.2989*r + 0.5870*g + 0.1140*b)
    alpha[alpha < thresh] = 0.0
    return alpha.astype(np.float32)

# def derive_fixed_mask(H: int, W: int, wm_path: str, alpha_thresh: float) -> np.ndarray:
#     a = _wm_alpha_fullwidth(wm_path, out_width=W, thresh=alpha_thresh)
#     mask = np.zeros((H, W), dtype=np.float32)
#     hh, ww = min(H, a.shape[0]), min(W, a.shape[1])
#     mask[0:hh, 0:ww] = (a[:hh, :ww] > 0).astype(np.float32)
#     return mask

def derive_fixed_mask(H: int, W: int, wm_path: str, alpha_thresh: float) -> np.ndarray:
    # Resize to full width, compute alpha in [0,1], zero out near-white background
    a = _wm_alpha_fullwidth(wm_path, out_width=W, thresh=alpha_thresh)  # uncropped
    # Paste at (0,0) without trimming (crop only if banner taller than image)
    alpha_full = np.zeros((H, W), dtype=np.float32)
    hh, ww = min(H, a.shape[0]), min(W, a.shape[1])
    if hh > 0 and ww > 0:
        alpha_full[0:hh, 0:ww] = a[:hh, :ww]
    # IMPORTANT: match generator’s mask rule: (alpha_full > alpha_thresh)
    mask = (alpha_thresh is not None) and (alpha_full > float(alpha_thresh))
    return mask.astype(np.float32)


def _prune_flatten_identity_inplace(mod: nn.Module):
    """
    Recursively rebuild nn.Sequential children to exclude nn.Flatten / nn.Identity.
    This removes those modules from the graph so Zennit/Captum never see them.
    """
    # First, filter any nn.Sequential children
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.Sequential):
            # Filter out Flatten / Identity from this Sequential
            kept = [m for m in child if not isinstance(m, (nn.Flatten, nn.Identity))]
            new_seq = nn.Sequential(*kept)
            setattr(mod, name, new_seq)
            
    # Then recurse into all children
    for name, child in list(mod.named_children()):
        _prune_flatten_identity_inplace(child)

class ForwardNoFlatten(nn.Module):
    """
    Wrap a trained model and execute forward without any Flatten/Identity modules.
    Supports:
      - original Net with layer1/layer3/layer5 + fc/fc1/fc2
      - features/head style nets where head originally began with Flatten
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        # Physically remove Flatten/Identity from the module graph
        _prune_flatten_identity_inplace(base)
        self.base = base

        if hasattr(base, "features") and hasattr(base, "head") and isinstance(base.head, nn.Sequential):
            self.mode = "feat_head"
        elif all(hasattr(base, a) for a in ("layer1", "layer3", "layer5", "fc", "fc1", "fc2")):
            self.mode = "classic"
        else:
            self.mode = "fallback"  # will just call base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "feat_head":
            x = self.base.features(x)
            x = x.reshape(x.size(0), -1)   # tensor op; no module
            x = self.base.head(x)          # head has been pruned of Flatten/Identity
            return x

        if self.mode == "classic":
            x = self.base.layer1(x)
            x = self.base.layer3(x)
            x = self.base.layer5(x)
            x = x.reshape(x.size(0), -1)   # tensor op; no module
            x = self.base.fc(x)
            x = self.base.fc1(x)
            x = self.base.fc2(x)
            return x

        return self.base(x)

def load_model(models_dir, base, split_index, scale, position, invert, seed_value, device):
    from watermarks.train_watermarks_server import Net  # or train_cnn.Net, whichever you trained
    path = model_path(models_dir, base, split_index, scale, position, invert, seed_value)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = Net().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model = ForwardNoFlatten(model)   # <- wrap AFTER loading
    return model, path


class Explainers:
    def __init__(self, model: nn.Module):
        self.ig = IntegratedGradients(model)
        self.gs = GradientShap(model)
        self.de = Deconvolution(model)
        self.lrp = LRP(model)

    def run_all(self, x: torch.Tensor, target: int, device) -> dict[str, np.ndarray]:
        def cap(attr):
            arr = attr.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            return combine_attr_rgb_mean_abs(arr)
        return {
            "int_grads": np.abs(cap(self.ig.attribute(x, target=target))),
            "shap":      np.abs(cap(self.gs.attribute(x, target=target, baselines=torch.zeros_like(x, device=device)))),
            "deconv":    np.abs(cap(self.de.attribute(x, target=target))),
            "lrp":       np.abs(cap(self.lrp.attribute(x, target=target))),
        }

def lrp_alpha_beta_abs(x: torch.Tensor, model: nn.Module, target: int, device) -> np.ndarray:
    z = x.clone().detach().requires_grad_(True).to(device)
    comp = EpsilonAlpha2Beta1()
    with comp.context(model) as mod:
        out = mod(z); grad = torch.eye(out.shape[1], device=device)[[target]]
        out.backward(gradient=grad)
    att = z.grad.detach().cpu().numpy()[0].transpose(1, 2, 0)
    return combine_attr_rgb_mean_abs(att)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-index", type=int, required=True)
    ap.add_argument("--seed-index", type=int, required=True)
    ap.add_argument("--position", choices=["fixed","variable"], default="fixed")
    ap.add_argument("--scale", choices=["zero_one","neg_one_one"], default="zero_one")
    ap.add_argument("--invert", type=int, default=0)

    ap.add_argument("--mask-mode", choices=["auto","derive-fixed","provided-only"], default="auto")
    ap.add_argument("--watermark", type=str, default="./watermark banner.jpg")
    ap.add_argument("--alpha-thresh", type=float, default=5/255)

    ap.add_argument("--artifacts-dir", type=str, default="./artifacts")
    ap.add_argument("--models-dir", type=str, default="./models")
    ap.add_argument("--energies-dir", type=str, default="./energies")
    ap.add_argument("--explanations-dir", type=str, default="./explanations")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args, _ = ap.parse_known_args(argv)

    invert_bool = bool(args.invert)
    device = get_device(args.gpu)
    seed_value = SEEDS[args.seed_index]

    (X_wm, y_wm, masks_wm), (X_no, y_no), p_all, _ = load_test_sets(
        args.artifacts_dir, args.split_index, args.scale, args.position, invert_bool
    )

    # decide mask source
    use_provided = masks_wm is not None and np.asarray(masks_wm).ndim >= 2
    if args.mask_mode == "provided-only" and not use_provided:
        raise ValueError(f"{p_all} has no masks and mask-mode=provided-only")
    if args.mask_mode == "derive-fixed" or (args.mask_mode == "auto" and not use_provided):
        if args.position != "fixed":
            raise ValueError("mask derivation requires --position fixed")
        H, W = X_wm.shape[2], X_wm.shape[3]
        fixed_mask = derive_fixed_mask(H, W, args.watermark, args.alpha_thresh)
        masks_wm = [fixed_mask] * len(X_wm)

    wm_avg = X_wm[: (args.limit or len(X_wm))].mean(axis=0)
    no_avg = X_no[: (args.limit or len(X_no))].mean(axis=0)

    model_conf, p_conf = load_model(args.models_dir, "confounder", args.split_index, args.scale, args.position, invert_bool, seed_value, device)
    model_sup,  p_sup  = load_model(args.models_dir, "suppressor", args.split_index, args.scale, args.position, invert_bool, seed_value, device)
    model_no,   p_no   = load_model(args.models_dir, "no_watermark", args.split_index, args.scale, args.position, invert_bool, seed_value, device)

    exp_conf, exp_sup, exp_no = Explainers(model_conf), Explainers(model_sup), Explainers(model_no)

    methods = ["deconv","int_grads","shap","lrp","lrp_ab"]
    def z(): return {m: [] for m in methods}
    energy_water_conf, energy_water_sup, energy_water_no = z(), z(), z()
    energy_no_water_conf, energy_no_water_sup, energy_no_water_no = z(), z(), z()
    explanations_water_conf, explanations_water_sup, explanations_water_no = z(), z(), z()
    explanations_no_water_conf, explanations_no_water_sup, explanations_no_water_no = z(), z(), z()
    res_baselines = {"x":[[],[]], "sobel":[[],[]], "laplace":[[],[]]}

    N = len(X_wm) if args.limit is None else min(args.limit, len(X_wm))
    rgb_w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    print(f"Device {device} | N={N}")
    for i in range(N):
        w_img = X_wm[i]; nw_img = X_no[i]; w_mask = np.asarray(masks_wm[i])
        w_ex = torch.from_numpy(w_img[None]).to(device, dtype=torch.float32)
        nw_ex= torch.from_numpy(nw_img[None]).to(device, dtype=torch.float32)
        with torch.no_grad():
            t_conf, t_sup, t_no = *(torch.argmax(model_conf(w_ex),1).item(),), *(torch.argmax(model_sup(w_ex),1).item(),), *(torch.argmax(model_no(w_ex),1).item(),)
            t_conf_nw, t_sup_nw, t_no_nw = *(torch.argmax(model_conf(nw_ex),1).item(),), *(torch.argmax(model_sup(nw_ex),1).item(),), *(torch.argmax(model_no(nw_ex),1).item(),)

        a_conf_w = exp_conf.run_all(w_ex, t_conf, device); a_conf_w["lrp_ab"] = lrp_alpha_beta_abs(w_ex, model_conf, t_conf, device)
        a_sup_w  = exp_sup.run_all(w_ex,  t_sup,  device); a_sup_w["lrp_ab"]  = lrp_alpha_beta_abs(w_ex,  model_sup,  t_sup,  device)
        a_no_w   = exp_no.run_all(w_ex,   t_no,   device); a_no_w["lrp_ab"]   = lrp_alpha_beta_abs(w_ex,   model_no,   t_no,  device)

        a_conf_nw= exp_conf.run_all(nw_ex,t_conf_nw,device); a_conf_nw["lrp_ab"]= lrp_alpha_beta_abs(nw_ex, model_conf, t_conf_nw, device)
        a_sup_nw = exp_sup.run_all(nw_ex, t_sup_nw, device); a_sup_nw["lrp_ab"] = lrp_alpha_beta_abs(nw_ex,  model_sup,  t_sup_nw, device)
        a_no_nw  = exp_no.run_all(nw_ex,  t_no_nw, device); a_no_nw["lrp_ab"]  = lrp_alpha_beta_abs(nw_ex,   model_no,   t_no_nw, device)

        for m in methods:
            energy_water_conf[m].append(energy(a_conf_w[m], w_mask))
            energy_water_sup[m].append(energy(a_sup_w[m], w_mask))
            energy_water_no[m].append(energy(a_no_w[m], w_mask))
            energy_no_water_conf[m].append(energy(a_conf_nw[m], w_mask))
            energy_no_water_sup[m].append(energy(a_sup_nw[m], w_mask))
            energy_no_water_no[m].append(energy(a_no_nw[m], w_mask))
            explanations_water_conf[m].append(a_conf_w[m])
            explanations_water_sup[m].append(a_sup_w[m])
            explanations_water_no[m].append(a_no_w[m])
            explanations_no_water_conf[m].append(a_conf_nw[m])
            explanations_no_water_sup[m].append(a_sup_nw[m])
            explanations_no_water_no[m].append(a_no_nw[m])

        def intensity(img): return np.tensordot(img.transpose(1,2,0)[...,:3], rgb_w, axes=([2],[0]))
        x_wm = energy(intensity(w_img), w_mask); x_no = energy(intensity(nw_img), w_mask)
        def sobel_laplace(img, mean_img):
            s = (img - mean_img).transpose(1,2,0)
            r,g,b = s[...,0], s[...,1], s[...,2]
            return (np.abs(laplace(r))+np.abs(laplace(g))+np.abs(laplace(b)),
                    np.abs(sobel(r))+np.abs(sobel(g))+np.abs(sobel(b)))
        lap_wm, sob_wm = sobel_laplace(w_img, wm_avg)
        lap_no, sob_no = sobel_laplace(nw_img, no_avg)
        res_baselines["laplace"][0].append(energy(lap_wm, w_mask))
        res_baselines["laplace"][1].append(energy(lap_no, w_mask))
        res_baselines["sobel"][0].append(energy(sob_wm, w_mask))
        res_baselines["sobel"][1].append(energy(sob_no, w_mask))
        res_baselines["x"][0].append(x_wm); res_baselines["x"][1].append(x_no)
        if i % 100 == 0: print(f"{i}/{N}  {time.time():.0f}")

    for b, (wm_list, no_list) in res_baselines.items():
        for d in (energy_water_conf, energy_water_sup, energy_water_no): d[b] = wm_list
        for d in (energy_no_water_conf, energy_no_water_sup, energy_no_water_no): d[b] = no_list

    os.makedirs(args.energies_dir, exist_ok=True); os.makedirs(args.explanations_dir, exist_ok=True)
    suf = suffix_for(args.scale, args.position, invert_bool); split = args.split_index; seed_tag = f"seed{seed_value}"
    def save(obj, kind, base):
        path = os.path.join(args.energies_dir if kind=="energy" else args.explanations_dir,
                            f"{kind}_{base}_pred{suf}_split{split}_{seed_tag}.pickle")
        with open(path, "wb") as f: pickle.dump(obj, f)
        print("saved:", path)

    save(energy_water_conf, "energy", "water_conf");  save(energy_water_sup,  "energy", "water_sup");  save(energy_water_no, "energy", "water_no")
    save(energy_no_water_conf, "energy", "no_water_conf"); save(energy_no_water_sup, "energy", "no_water_sup"); save(energy_no_water_no, "energy", "no_water_no")
    save(explanations_water_conf, "explanations", "water_conf"); save(explanations_water_sup, "explanations", "water_sup"); save(explanations_water_no, "explanations", "water_no")
    save(explanations_no_water_conf, "explanations", "no_water_conf"); save(explanations_no_water_sup, "explanations", "no_water_sup"); save(explanations_no_water_no, "explanations", "no_water_no")

if __name__ == "__main__":
    main()
