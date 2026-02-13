class EarlyStop:
    """
    An early stopping handler to make setting up early stopping easier
    """
    def __init__(self, patience=5, min_delta=0.0, mode="loss"):
        """
        Initializes the early stopping handler.
        
        :param patience: Number of epochs to wait for improvement before stopping
        :param min_delta: Minimum change in validation loss to qualify as an improvement
        :param mode: "loss" to monitor validation loss, "accuracy" to monitor validation accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_model_state = None

    def __call__(self, score, model):
        """
        Checks if training should be stopped
        """
        # Setting up the best_score if not setup yet
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            return False
        
        # Checking if the score improved
        if self.mode == "loss":
            improved = score < self.best_score - self.min_delta
        else:  # mode == "accuracy"
            improved = score > self.best_score + self.min_delta

        if improved:
            # If improved, reset counter and save best model state
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict()
            return False
        else:
            # If not improved, increment counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def get_best_model_state(self):
        """
        Returns the best model state saved during training
        """
        return self.best_model_state
