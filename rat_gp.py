import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available as cuda_available, empty_cache
from SALib.sample import saltelli
from SALib.analyze import sobol
from os.path import join
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")


class GPRegressionModel(ExactGP):
    """
    The gpytorch model underlying the Rat_GP class.
    """
    def __init__(self, train_x, train_y, likelihood):
        """
        Constructor that creates objects necessary for evaluating GP.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # The Mattern Kernal is particularly well suited to models with abrupt transitions between success and failure.
        self.covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=13))

    def forward(self, x):
        """
        Takes in nxd data x and returns a MultivariateNormal with the prior mean and covariance evaluated at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class Rat_GP():
    """
    Class for the Rat GP model.
    """
    def __init__(self, training_data, testing_data=None, model_type="composite"):
        data = pd.read_csv(join("data", training_data))
        self.resistance_simulated = False
        if sum(data["RESISTANCE_RATE"]):
            self.resistance_simulated = True
        self.testing_data = testing_data
        data = data.to_numpy(dtype="float")
        if data[0][0]:
            self.drive_type = "Viability homing"
        elif data[0][1]:
            self.drive_type = "Female fertility homing"
        elif data[0][2]:
            self.drive_type = "Y-shredder"
        else:
            raise ValueError("No gene drive?")

        # The model input parameters in the training set.
        self.train_x = torch.from_numpy(data[:,3:16]).float().contiguous()

        self.model_type  = model_type
        if self.model_type  == "composite":
            # GP trained on the composite value.
            self.train_y = torch.from_numpy(data[:,23:24]).float().contiguous().flatten()
            self.y_noise = torch.from_numpy(data[:,24:25]).float().contiguous().flatten()
            if cuda_available():
                self.y_noise = self.y_noise.cuda()
            self.likelihood = FixedNoiseGaussianLikelihood(self.y_noise, learn_additional_noise=False)
        elif self.model_type  == "suppression_rate":
            # GP trained on the percent of runs at each data point that suppression took place.
            self.train_y = torch.from_numpy(data[:,21:22]).float().contiguous().flatten()
            self.likelihood = GaussianLikelihood()
        else:
            raise ValueError("Specify a model_type of \"composite\" or \"suppression_rate\".")

        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood)

        if cuda_available():
            self.train_x, self.train_y, self.likelihood, self.model = self.train_x.cuda(), self.train_y.cuda(), self.likelihood.cuda(), self.model.cuda()

        min_drive_quality = 0.75
        if self.drive_type == "Viability homing" or self.drive_type == "Female fertility homing" and not self.resistance_simulated:
            min_drive_quality = 0.5

        self.default_params = {
                'Density': 1000.0,
                'Island side length': 2.0,
                'Interaction distance': 75.0,
                'Avg. dispersal': 250.0,
                'Monthly survival rate': 0.9,
                'Litter size': 4.0,
                'Itinerant frequency': 0.1,
                'Itin. dispersal multiplier': 2.0,
                'Release Percentage': 0.1,
                'Drive fitness': 1.0,
                'Drive efficiency': 0.9,
                'Resistance rate': 0.0,
                'R1 rate': 0.0
        }
        self.param_ranges = {
                "Density" : (600, 1500),
                "Island side length" : (1, 5),
                "Interaction distance" : (60, 300),
                "Avg. dispersal" : (25, 1000),
                "Monthly survival rate" : (0.7, 0.95),
                "Litter size" : (2, 8),
                "Itinerant frequency" : (0, 0.5),
                "Itin. dispersal multiplier" : (1, 5),
                "Release Percentage" : (0.01, 0.5),
                "Drive fitness" : (min_drive_quality, 1.0),
                "Drive efficiency" :(min_drive_quality, 1.0),
                "Resistance rate" : (0.0, 0.1),
                "R1 rate" : (0.0, 0.02)
        }
        # Don't include resistance parameters in the SA if the model doesn't simulate resistance:
        num_params = 11
        if self.resistance_simulated:
            num_params = 13
        self.sa_params_dict = {
                "num_vars": num_params,
                "names": [k for k, v in self.param_ranges.items()][:num_params],
                "bounds": [v for k, v in self.param_ranges.items()][:num_params]
        }

    def save(self, filename, cpu=False):
        """
        Saves the trained GP model.
        """
        model_to_save = self.model
        if cpu:
            model_to_save = self.model.cpu()
        torch.save(model_to_save.state_dict(), f"{filename}.pth")
        print("Model saved.")

    def load(self, filename):
        """
        Loads a pre-trained GP model.
        """
        try:
            self.model.load_state_dict(torch.load(f"{filename}.pth"))
        except FileNotFoundError:
            try:
                self.model.load_state_dict(torch.load(filename))
            except FileNotFoundError:
                raise FileNotFoundError(f"{filename} not found.")
        # Torch requires that a loaded model be "retrained" on the training data.
        self.train(10)
        print("Model loaded.")

    def train(self, num_iterations):
        """
        Train the model.
        """
        # Set the model to training mode.
        self.model.train()
        self.likelihood.train()
        # Using the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)
        # "Loss" for GPs: the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        if num_iterations >= 100:
            print(f"Training for {num_iterations} iterations:")
        # Training loop:
        with gpytorch.settings.max_cg_iterations(16000):
            for i in range(num_iterations):
                # Zero gradients from previous iteration:
                optimizer.zero_grad()
                # Output from model:
                output = self.model(self.train_x)
                # Calc loss and backprop gradients:
                loss = -mll(output, self.train_y)
                loss.backward()
                if (i+1) % 100 == 0:
                    print(f"Iter {i + 1}/{num_iterations} - Loss: {loss.item()}")
                optimizer.step()
                empty_cache()
        # Set the model to evaluation mode.
        self.model.eval()
        self.likelihood.eval()

    def predict(self, data, drive_cols=True):
        """
        Predicts y values, lower, and upper confidence for a data set.
        Takes data of the form of a panda dataframe
        drive_cols is set to false when the dataset does not have three true/false columns for drive type.
        """
        data = data.to_numpy(dtype="float")
        if drive_cols:
            x = torch.from_numpy(data[:,3:16]).float().contiguous()
        else:
            x = torch.from_numpy(data[:,:13]).float().contiguous()
        if cuda_available():
            x = x.cuda()
        return self.predict_ys(x)

    def predict_ys(self, parsed_data):
        """
        Predicts y values from X values.
        Takes parsed data as a contiguous (cuda if available) torch tensor.
        """
        loader = DataLoader(TensorDataset(parsed_data), batch_size=1024, shuffle=False)
        mean, lower, upper = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch in loader:
                observed_pred = self.likelihood(self.model(batch[0]))
                cur_mean = observed_pred.mean
                if cuda_available():
                    mean = torch.cat([mean, cur_mean.cpu()])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower.cpu()])
                    upper = torch.cat([upper, cur_upper.cpu()])
                else:
                    mean = torch.cat([mean, cur_mean])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower])
                    upper = torch.cat([upper, cur_upper])
        return mean[1:], lower[1:], upper[1:]

    def check_accuracy(self, data=None):
        """
        Check errors between predicted values and known result values.
        """
        if data == None:
            data = self.testing_data
            if data == None:
                raise ValueError("Specify a test set to check.")
        data = pd.read_csv(join("data", data))
        predicted_mean, _, _ = self.predict(data)
        data = data.to_numpy(dtype="float")
        target = torch.from_numpy(data[:,21:22]).float().contiguous().flatten()
        if self.model_type  == "composite":
            # Shift predictions from [-1, 1] to [0, 1].
            predicted_mean += 1
            predicted_mean /= 2

        rmse = sum(np.sqrt((target.numpy() - predicted_mean.numpy())**2) / len(target))
        actual_success = 0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        for i in range(len(predicted_mean)):
            if target[i] >= 0.5:
                actual_success += 1
            if predicted_mean[i] >= 0.5 and target[i] >= 0.5:
                true_positives += 1
            if predicted_mean[i] < 0.5 and target[i] >= 0.5:
                false_negatives += 1
            if predicted_mean[i] >= 0.5 and target[i] < 0.5:
                false_positives += 1
        precision = 0
        recall = 0
        if true_positives + false_positives:
            precision = true_positives / (true_positives + false_positives)
        if actual_success:
            recall = true_positives / actual_success
        print(f"Number of suppressions in data: {actual_success}")
        print(f"True positives: {true_positives}")
        print(f"False negatives: {false_negatives}")
        print(f"False positives: {false_positives}")
        print(f"Total number of erroneous predictions: {false_negatives + false_positives} out of {len(target)}.")
        print(f"RMSE: {rmse}\nPrecision: {precision}\nRecall: {recall}")

    def sensitivity_analysis(self, base_sample=10000, param_ranges=None, verbose=False):
        """
        Perform a sensitivity analysis. Print the analysis if verbose=True.
        Returns a list of 3 pandas dataframes, where
        the first entry in the list is total effects, the second entry is first order, and the third entry is second order effects.
        """
        sa_params = deepcopy(self.sa_params_dict)
        if param_ranges:
            for key in param_ranges:
                if key not in self.default_params:
                    print(f"\"{key}\" not a valid parameter name. Ignoring.")
                if key in sa_params["names"]:
                    sa_params["bounds"][sa_params["names"].index(key)] = param_ranges[key]

        for i in range(len(sa_params["bounds"])):
            if type(sa_params["bounds"][i]) is not list and type(sa_params["bounds"][i]) is not tuple:
                sa_params["bounds"][i] = (sa_params["bounds"][i], sa_params["bounds"][i] + 0.00000001)

        # Inconveniently, the GPs were trained on sigma, instead of actual interaction distance, so we need to divide by 3 before the GP can predict points.
        sa_params["bounds"][sa_params["names"].index("Interaction distance")] = (sa_params["bounds"][sa_params["names"].index("Interaction distance")][0] / 3,
                                                                                 sa_params["bounds"][sa_params["names"].index("Interaction distance")][1] / 3)

        # Generate samples
        param_values = saltelli.sample(sa_params, base_sample)
        # Evaluate the model at sampled points:
        x = np.zeros((len(param_values), 13))
        if self.resistance_simulated:
            for i in range(len(param_values)):
                x[i] = param_values[i]
        else:
            for i in range(len(param_values)):
                x[i] = np.append(param_values[i], [0, 0])
        x = torch.from_numpy(x).float().contiguous()
        if cuda_available():
            x = x.cuda()
        y, _, _ = self.predict_ys(x)
        y = y.numpy()
        # Perform the sensitivity analysis:
        sa = sobol.analyze(sa_params, y, print_to_console=verbose)
        sa_df = sa.to_df()
        sa_df[0].columns = [c.replace('ST', 'Total Effects') for c in sa_df[0].columns]
        sa_df[1].columns = [c.replace('S1', 'First Order') for c in sa_df[1].columns]
        sa_df[2].columns = [c.replace('S2', 'Second Order') for c in sa_df[2].columns]
        sa_df.append(f"{self.drive_type}{' with resistance' if self.resistance_simulated else ' without resistance'}")
        return sa_df


def load_model(model_name, model_type, force_cpu=False):
    """
    Load a pretrained model along with the data used to train and test it.
    """
    if model_name not in ["viability_homing", "viability_homing_resistance", "female_fertility_homing", "female_fertility_homing_resistance", "y_shredder"]:
        raise ValueError("Specify a model_name of \"viability_homing\", \"viability_homing_resistance\", \"female_fertility_homing\", \"female_fertility_homing_resistance\", \"y_shredder\".")
    if model_type not in ["composite", "suppression_rate"]:
        raise ValueError("Specify a model_type of \"composite\" or \"suppression_rate\".")
    save_name = f"{model_name}_{model_type}{'_cpu' if not cuda_available() or force_cpu else ''}"
    training_dataset = f"{model_name.upper()}_TRAIN.csv"
    test_dataset = f"{model_name.upper()}_TEST.csv"
    model = Rat_GP(training_dataset, test_dataset, model_type=model_type)
    model.load(join("models", save_name))
    return model
