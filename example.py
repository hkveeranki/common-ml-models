import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from neuralnetworks.activationfunctions.tanh import TanH
from neuralnetworks.layer import Layer
from neuralnetworks.multilayer_neural_network import MultiLayerNeuralNetwork
from neuralnetworks.activationfunctions.relu import ReLU
from neuralnetworks.activationfunctions.sigmoid import Sigmoid

print('Reading data')
df = pd.read_csv('small.csv')
print('Done')
# df = df.head(2000)
print(df.label.unique())
num_labels = len(df.label.unique())
y_app = df.label.values
X_app = df.iloc[:, :].drop('label', axis=1)
# One hot encode the labels
print('One hot encoding the labels')
one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
print('Done.')
print('Performing test val split.')
x_train, x_val, y_train, y_val = train_test_split(X_app, y_app,
                                                  test_size=0.2,
                                                  random_state=42)
integer_encoded = y_train.reshape(len(y_train), 1)
y_train = one_hot_encoder.fit_transform(integer_encoded)
y_val = one_hot_encoder.fit_transform(y_val.reshape(len(y_val), 1))
x_train = x_train.values
scaler = preprocessing.StandardScaler().fit(x_train)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = x_val.values
x_val = scaler.transform(x_val)

print('Done.')
input_len = len(x_train[0])
nn = MultiLayerNeuralNetwork(n_input=input_len, n_output=num_labels)
sigmoid = Sigmoid()
relu = ReLU()
tanh = TanH()
nn.add(Layer(size=512, activation_function=sigmoid))
nn.add(Layer(size=128, activation_function=tanh))
# nn.add(Layer(size=64, activation_function=relu))
# nn.add(Layer(size=32, activation_function=tanh))
nn.compile(output_activation_function=sigmoid)
print('Neural network compiled.')
print('Starting the training.')
nn.train(x_train, y_train, eta=0.001, n_epochs=1000, x_val=x_val, y_val=y_val,
         batch_size=64)
print('Training complete.')
nn.save_weights('weights.pickle')
