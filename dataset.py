import subprocess
import sys
import threading
from pathlib import Path
from typing import BinaryIO, Iterator

import torch
from torch.utils.data import IterableDataset


def read_tensor_from_file(
    file: BinaryIO, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = num_elements * bytes_per_element

    buffer = bytearray()
    while len(buffer) < total_bytes:
        bytes_remaining = total_bytes - len(buffer)
        chunk = file.read(bytes_remaining)
        if not chunk:
            if len(buffer) == 0:
                raise EOFError("End of file reached.")
            else:
                raise ValueError(
                    f"Expected {total_bytes} bytes, but only received {len(buffer)} bytes before EOF."
                )
        buffer.extend(chunk)

    data = bytes(buffer)
    tensor = torch.frombuffer(data, dtype=dtype)
    tensor = tensor.reshape(shape)
    return tensor


class ChessDatasetClient(IterableDataset):
    def __init__(
        self,
        server_binary: Path,
        data_folder: Path,
        threads: int,
        queue_size: int,
    ):
        self.server_binary = server_binary

        self.process = subprocess.Popen(
            [
                str(self.server_binary),
                "data-server",
                str(data_folder),
                str(threads),
                str(queue_size),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,  # 1MB buffer size
        )

        if self.process.stdout is None:
            raise RuntimeError("Failed to capture stdout from Rust binary.")

        self.pipe = self.process.stdout

        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

    def _read_stderr(self):
        for line in self.process.stderr:
            sys.stderr.buffer.write(line)
            sys.stderr.flush()

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:
            try:
                state = read_tensor_from_file(self.pipe, torch.bfloat16, (119, 8, 8))
                legal_moves_mask = read_tensor_from_file(
                    self.pipe, torch.int8, (73 * 8 * 8,)
                )
                policy = read_tensor_from_file(self.pipe, torch.int64, (1,))
                value = read_tensor_from_file(self.pipe, torch.int64, (1,))
                yield state, legal_moves_mask.to(torch.bool), policy[0], value[0]
            except (EOFError, ValueError) as e:
                print(f"Error reading data from Rust binary: {e}", file=sys.stderr)
                break

    def __del__(self):
        if hasattr(self, "process"):
            self.process.terminate()
            self.process.wait()


def main() -> None:
    from tqdm import tqdm

    dataset = ChessDatasetClient(
        server_binary=Path("/home/tanmay/chester/target/release/slchess"),
        data_folder=Path("/home/tanmay/chess-ntp/pgns"),
        threads=6,
        queue_size=2048,
    )

    for idx, sample in enumerate(tqdm(dataset)):
        if idx == 0:
            for x in sample:
                print(x)
        pass


if __name__ == "__main__":
    main()
