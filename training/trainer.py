import sys
sys.path.append("third_party/colbert_v1/ColBERT_v1")

from infra.run import Run
from infra.launcher import Launcher
from infra.config import ColBERTConfig, RunConfig

from training.training import train

class Trainer:
    def __init__(self, triples, rag_collection_path, rag_dataset_path, config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.rag_collection_path = rag_collection_path
        self.rag_dataset_path = rag_dataset_path

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint="bert-base-uncased", n_epochs: int=1):
        """
        Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.rag_collection_path, collection=self.rag_dataset_path)
        self.configure(checkpoint=checkpoint)

        launcher = Launcher(train)

        self._best_checkpoint_path = launcher.launch(
            self.config, self.triples, self.rag_collection_path, self.rag_dataset_path, n_epochs
        )

    def best_checkpoint_path(self):
        return self._best_checkpoint_path
