class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layer_num):
		self.model = model
		self.gradients = []

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):


		
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output


class GradCam:
	def __init__(self, model, target_layer_num):
		self.model = model
		self.extractor = ModelOutputs(self.model, target_layer_num)

	def forward(self, in_var):
		return self.model(in_var)

	def __
