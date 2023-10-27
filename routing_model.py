import collections
import os
import torch


class Routing(torch.nn.Module):
    """
    Implements routing algorithm.
    Takes in input tokens, embedding_layer and outputs which model to route to.
    """
    def __init__(self, embedding_layer, routing_attention_layer, n_models=2):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.routing_attention_layer = routing_attention_layer
        input_size = self.embedding_layer.weight.shape[-1]

        n_inter_1 = input_size // 2
        self.routing_layer_1 = torch.nn.Linear(input_size, n_inter_1)
        n_inter_2 = n_inter_1 // 2
        self.routing_layer_2 = torch.nn.Linear(n_inter_1, n_inter_2)
        self.routing_layer_3 = torch.nn.Linear(n_inter_2, n_models)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.output = collections.namedtuple("Output", ["loss", "logits", "top_gate"])

    def forward(self, input_ids=None, labels=None):
        """
        Takes in input string and outputs which model to route to.

        Parameters
        ----------
        input_string: str
            Input string to be routed
        labels: list
            List of labels to be routed to
        """
        assert input_ids is not None, "Input ids must be provided"
        if self.training:
            assert labels is not None, "Labels must be provided during training"

        input_embeddings = self.embedding_layer(input_ids)
        input_embeddings = self._convert_seq_to_single_feature(input_embeddings)

        loss, logits, top_gate = self.top_k_gating(
            input_embeddings, self.training, labels=labels
        )

        return self.output(loss=loss, logits=logits, top_gate=top_gate)

    def _convert_seq_to_single_feature(self, x):
        """
        Flatten the second and third dimensions of a Tensor.
        """
        out = self.routing_attention_layer(x)
        if isinstance(out, tuple):
            out = out[0]

        return out.mean(dim=1)

    def top_k_gating(self, x, train, labels=None):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        with torch.autocast(device_type=x.device.type):
            inter_1 = self.relu(self.routing_layer_1(x))
            inter_2 = self.relu(self.routing_layer_2(inter_1))
            clean_logits = self.routing_layer_3(inter_2)
            logits = clean_logits

        if len(logits.shape) > 2:
            logits = logits.squeeze()

        top_k_gates = self.softmax(logits)

        loss = None
        top_gate = None
        if train:
            # Compute loss
            loss = self.ce_loss(top_k_gates, labels)
        else:
            top_gate = top_k_gates.argmax(dim=-1)

        return loss, top_k_gates, top_gate


def load_model(path):
    """
    Loads model from path.
    """
    # 1. Load embedding_fn and attention_fn
    embedding_fn = torch.load(os.path.join(path, "embedding_fn.pt"))
    attention_fn = torch.load(os.path.join(path, "attention_fn.pt"))

    # 2. Load routing model
    routing = Routing(embedding_fn, attention_fn, n_models=4)
    routing.load_state_dict(torch.load(os.path.join(path, "routing.pt")))

    return routing
