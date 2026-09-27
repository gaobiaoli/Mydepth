from pathlib import Path

from model.scale_shift_dense2dense_noweight import PriorBIMDA
from train_dense2dense import main


CALIBRATION_CHECKPOINT = Path(
    __file__
).resolve().parent / Path(
    "outputs/train_onlyscale_shift_area2-6-s42/latest.pt"
)


class TrainingModel(PriorBIMDA):
    @classmethod
    def from_pretrained(cls, local_files_only=False):
        return super().from_pretrained(
            local_files_only=local_files_only,
            calibration_checkpoint=CALIBRATION_CHECKPOINT,
        )


if __name__ == "__main__":
    main(TrainingModel)
