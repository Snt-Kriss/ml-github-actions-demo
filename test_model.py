from model import train_and_save_model

def test_training_accuracy():
    acc= train_and_save_model()
    assert acc > 0.7, "Model accuracy is too low!"