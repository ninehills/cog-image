import torch
import json
import time

from cog import BasePredictor, Input, BaseModel
from typing import List
from FlagEmbedding import FlagReranker

class Output(BaseModel):
    """Output schema for the predictor"""
    scores: List[float]
    model_name: str
    use_fp16: bool


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_name = "bge-reranker-large"
        self.use_fp16 = torch.cuda.is_available()
        self.model = FlagReranker(f"checkpoints", use_fp16=self.use_fp16)

    def predict(self,
        pairs_json: str = Input(description='Input pairs, eg: [["a", "b"], ["c", "d"]]'),
    ) -> Output:
        """Run a single prediction on the model"""
        inputs = json.loads(pairs_json)
        all_scores = self.model.compute_score(inputs)
        if isinstance(all_scores, list):
            scores = all_scores
        else:
            scores = [all_scores]
        return Output(scores=scores, model_name=self.model_name, use_fp16=self.use_fp16)
