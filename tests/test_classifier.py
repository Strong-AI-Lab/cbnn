

import pytest

from cbnn.model.modules.classifiers import MLPClassifier, BayesianClassifier

import torch


class TestMLPClassifier:
    @pytest.fixture
    def classifier(self):
        model = MLPClassifier(in_dim=10, out_dim=5, hidden_dim=20, num_layers=3)
        model.eval()
        return model

    def test_forward(self, classifier):
        x = torch.randn(1, 10)
        y = classifier(x)
        assert y.shape == (1, 5)


class TestBayesianClassifier:
    @pytest.fixture
    def classifier(self):
        return BayesianClassifier(in_dim=10, out_dim=5, hidden_dim=20, num_layers=3)

    def test_forward(self, classifier):
        x = torch.randn(1, 10)
        y, in_weights, out_weights = classifier(x)
        assert y.shape == (1, 5)
        assert in_weights.shape == (10,20)
        assert out_weights.shape == (20,5)

    def test_sample_weights(self, classifier):
        mean = torch.randn(10, 20)
        log_var = torch.randn(10, 20)
        eps = torch.randn(10, 20)
        weights = classifier._sample_weights(mean, log_var, eps)
        assert weights.shape == (10, 20)
        assert torch.allclose(weights, mean + torch.exp(0.5 * log_var) * eps)

    def test_forward_given_eps(self, classifier):
        x = torch.randn(1, 10)
        eps_in = torch.randn(10, 20)
        eps_out = torch.randn(20, 5)
        y, in_weights, out_weights = classifier(x, eps_in, eps_out)
        assert y.shape == (1, 5)
        assert in_weights.shape == (10,20)
        assert out_weights.shape == (20,5)
        assert torch.allclose(in_weights, classifier.fc_in_mean.weight + torch.exp(0.5 * classifier.fc_in_log_var.weight) * eps_in)
        assert torch.allclose(out_weights, classifier.fc_out_mean.weight + torch.exp(0.5 * classifier.fc_out_log_var.weight) * eps_out)
        
