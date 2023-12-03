from datetime import datetime
import torch


def filename_datetime() -> str:
	return datetime.now().strftime("%Y_%m_%d_T%H%M")


def get_device() -> torch.device:
	return torch.device(
		"cuda"
		if torch.cuda.is_available()
		else "mps"
		if torch.backends.mps.is_available()
		else "cpu"
	)
