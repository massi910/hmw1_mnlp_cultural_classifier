import torch


def resolve_device(device_cfg: str) -> torch.device:
    """
    Resolve runtime device from config.
    """
    device_cfg = device_cfg.lower()

    if device_cfg == "cpu":
        return torch.device("cpu")

    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError(f"Unknown device option: {device_cfg}")