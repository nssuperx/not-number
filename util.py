import torch


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        try:
            import torch_directml  # pyright: ignore[reportMissingImports]

            device = torch.device(torch_directml.device())
        except ImportError:
            pass

    return device
