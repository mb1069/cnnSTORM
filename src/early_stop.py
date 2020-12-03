from pytorch_lightning.callbacks import EarlyStopping


class EpochEarlyStop(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass
        # try:
        #     self._run_early_stopping_check(trainer, pl_module)
        # except RuntimeError as e:
        #     print(e)

    def on_train_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_epoch_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        # self._run_early_stopping_check(trainer, pl_module)
        try:
            self._run_early_stopping_check(trainer, pl_module)
        except RuntimeError as e:
            print(e)
