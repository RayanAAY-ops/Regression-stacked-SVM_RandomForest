import AE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
if __name__ == "__main__":
	all_data_encoded = np.load("sample/data.npy")


	model=AE.MyModel()
	model.compile(
	optimizer="adam",
	loss="mean_squared_error"  
	)
	num_epochs = 10000
	batch_size = 64

	history = model.fit(x=all_data_encoded, y=all_data_encoded,
					epochs=num_epochs,
					batch_size=batch_size,
					shuffle=True,
					
					verbose=1)

	np.save("sample/latent.npy",model.encode(all_data_encoded).numpy())