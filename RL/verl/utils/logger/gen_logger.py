# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from ..py_functional import is_package_available
import os

if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for inp, out, lab, score in samples:
            msg = f"[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n"
            print(msg)
            #self._write_to_file(msg, step, overwrite=False)

    '''def _write_to_file(self, text: str, step: int, overwrite: bool = False) -> None:
        self.log_path = './training_logs'
        mode = "w" if overwrite else "a"
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        save_path = os.path.join(self.log_path, f"{self.exp_name}_validation_outputs.log")
        with open(save_path, mode, encoding="utf-8") as f:
            f.write(f"\nStep {step}:\n")
            f.write(text.rstrip() + "\n")'''

@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


@dataclass
class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str]):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger]())

    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for logger in self.loggers:
            logger.log(samples, step)
