from typing import List
from rlbench.backend.task import Task
from typing import Tuple

class ReachBall(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        pass

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return ['']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]
