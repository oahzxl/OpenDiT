SKIP_MANAGER = None


class SkipManager:
    def __init__(
        self,
        steps: int = 100,
        cross_skip: bool = False,
        cross_threshold: int = 700,
        cross_gap: int = 5,
        spatial_skip: bool = False,
        spatial_threshold: int = 700,
        spatial_gap: int = 3,
        spatial_layer_range: list = [5, 27],
        temporal_skip: bool = False,
        temporal_threshold: int = 700,
        temporal_gap: int = 5,
    ):
        self.steps = steps

        self.cross_skip = cross_skip
        self.cross_threshold = cross_threshold
        self.cross_gap = cross_gap

        self.spatial_skip = spatial_skip
        self.spatial_threshold = spatial_threshold
        self.spatial_gap = spatial_gap
        self.spatial_layer_range = spatial_layer_range

        self.temporal_skip = temporal_skip
        self.temporal_threshold = temporal_threshold
        self.temporal_gap = temporal_gap

        print(
            f"Init SkipManager:\n\
            steps={steps}\n\
            cross_skip={cross_skip}, cross_threshold={cross_threshold}, cross_gap={cross_gap}\n\
            spatial_skip={spatial_skip}, spatial_threshold={spatial_threshold}, spatial_gap={spatial_gap}, spatial_layer_range={spatial_layer_range}\n\
            temporal_skip={temporal_skip}, temporal_threshold={temporal_threshold}, temporal_gap={temporal_gap}\n",
            end="",
        )

    def if_skip_cross(self, timestep: int, count: int):
        if (
            self.cross_skip
            and (timestep is not None)
            and (count % self.cross_gap != 0)
            and (self.cross_threshold[0] < timestep < self.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count

    def if_skip_temporal(self, timestep: int, count: int):
        if (
            self.temporal_skip
            and (timestep is not None)
            and (count % self.temporal_gap != 0)
            and (self.temporal_threshold[0] < timestep < self.temporal_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count

    def if_skip_spatial(self, timestep: int, count: int, block_idx: int):
        if (
            self.spatial_skip
            and (timestep is not None)
            and (count % self.spatial_gap != 0)
            and (self.spatial_threshold[0] < timestep < self.spatial_threshold[1])
            and (self.spatial_layer_range[0] <= block_idx <= self.spatial_layer_range[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count


def set_skip_manager(
    steps: int = 100,
    cross_skip: bool = False,
    cross_threshold: int = 700,
    cross_gap: int = 5,
    spatial_skip: bool = False,
    spatial_threshold: int = 700,
    spatial_gap: int = 3,
    spatial_block: list = [8, 27],
    temporal_skip: bool = False,
    temporal_threshold: int = 700,
    temporal_gap: int = 5,
):
    global SKIP_MANAGER
    SKIP_MANAGER = SkipManager(
        steps,
        cross_skip,
        cross_threshold,
        cross_gap,
        spatial_skip,
        spatial_threshold,
        spatial_gap,
        spatial_block,
        temporal_skip,
        temporal_threshold,
        temporal_gap,
    )


def enable_skip():
    if SKIP_MANAGER is None:
        return False
    return SKIP_MANAGER.cross_skip or SKIP_MANAGER.spatial_skip or SKIP_MANAGER.temporal_skip


def get_steps():
    return SKIP_MANAGER.steps


def get_cross_skip():
    return SKIP_MANAGER.cross_skip


def get_cross_threshold():
    return SKIP_MANAGER.cross_threshold


def get_cross_gap():
    return SKIP_MANAGER.cross_gap


def get_spatial_skip():
    return SKIP_MANAGER.spatial_skip


def get_spatial_threshold():
    return SKIP_MANAGER.spatial_threshold


def get_spatial_gap():
    return SKIP_MANAGER.spatial_gap


def get_spatial_layer_range():
    return SKIP_MANAGER.spatial_layer_range


def get_temporal_skip():
    return SKIP_MANAGER.temporal_skip


def get_temporal_threshold():
    return SKIP_MANAGER.temporal_threshold


def get_temporal_gap():
    return SKIP_MANAGER.temporal_gap


def if_skip_cross(timestep: int, count: int):
    return SKIP_MANAGER.if_skip_cross(timestep, count)


def if_skip_temporal(timestep: int, count: int):
    return SKIP_MANAGER.if_skip_temporal(timestep, count)


def if_skip_spatial(timestep: int, count: int, block_idx: int):
    return SKIP_MANAGER.if_skip_spatial(timestep, count, block_idx)
