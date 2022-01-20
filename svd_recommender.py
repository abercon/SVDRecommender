'''
DESCRIPTION:
This module exposes an implementation of a SVDRecommender model via the
SVDRecommender class.

DEPENDS ON:
numpy

PUBLIC CLASSES:
-------------------------------------------------------------------------------

class SVDRecommender(self, lmbda=0.5, n_factors = 100, latent_init = 0.1,
                     bias_init = 0):
    An implementation of a factor-based (SVD) collaborative filter. Call
    help(SVDRecommender) for more information.

'''

import numpy as np

class SVDRecommender():

    '''
    DESCRIPTION:
    An implementation of a factor-based (SVD) collaborative filter, based
    on:

    Koren, Y. and Bell, R. (2011) ‘Advances in Collaborative Filtering’,
    in Ricci, F. et al. (eds) Recommender Systems Handbook.
    Boston, MA: Springer US, pp. 145–186. doi: 10.1007/978-0-387-85820-3_5.

    Mean global rating μ, user and item biases and latents are all learnt
    using stochastic gradient descent. The two hyperparameters of the model
    are lambda, the regularisation weight(s) and n_factors, the number of
    latent factors used by the model.

    KEYWORD ARGUMENTS:
    -------------------------------------------------------------------------
    This class takes the following keyword arguments on instantiation:

    lmbda, of type float:
        The regularisation weight, typically this is low but it's value
        should be set based on cross validation. A vector of weights may be
        passed providing individual weights for regularisers of user_biases,
        item_biases, user_latents and item_latents. Must have shape 4 in
        this case.
    n_factors, of type int:
        The number of latent factors the model uses, set to 100 by default.

    ATTRIBUTES:
    ------------------------------------------------------------------------
    params, of type dict:
        A dictionary containing the parameters of the model. Has the
        following structure:
        {"user_latents":np.ndarray, shape [n_factors, n_users],
            "item_latents":np.ndarray, shape [n_factors, n_items],
            "user_bias":np.ndarray, shape[n_users],
            "item_bias":np.ndarray, shape[n_items],
            "mu": float,
            "n_factors": int,
            "lambda": np.ndarray, shape[4]}
    cost, of type float:
        The current value of the cost function as evaluated on the training
        data. Before fitting this is set to np.inf.

    PUBLIC METHODS:
    ------------------------------------------------------------------------
    def get_params(self) -> dict:
        Gets the parameters for the current state of the model

    fit(self, observed_ratings, learning_rate = 0.1, n_iter = 100) -> None:
        Fits the SVD recommender using stochasticc gradient descent. A
        differentapproach to optimisation was taken compared to
        (Ricci, F. et al, 2011),where only the sign of the gradient was
        used. Could potentially be improved by implementing an ADAM
        optimiser.

    predict(self) -> np.ndarray:
        Predicts the utility of each item for each contact based on the
        current fit of the model.

    PRIVATE METHODS:
    ------------------------------------------------------------------------
    __init__(self, lmbda=0.5, n_factors = 100, latent_init = 0.1,
             bias_init = 0):
        Initialises the model

    _cost(self, observed_ratings) -> float:
        Calculates the cost (sum of squared errors) of the model based on
        the current parameters and a set of observed ratings.

    _parse_hparams(self, param_input) -> np.ndarray:
        Takes hyperparameters and parses them to an array.
    '''

    def __init__(self, lmbda=0.5, n_factors = 100, latent_init = 0.1,
                 bias_init = 0):

        '''
        DESCRIPTION:
        Initialises the model.

        KEYWORD ARGUMENTS:
        -----------------------------------------------------------------------
        lmbda, of type float or iterable of floats with length 4:
            The regularisation weight(s) to use.
        n_factors, of type int:
            The number of latent factors to use.
        latent_init, of type float:
            The value with which to initialise the user and item latents.
        bias_init, of type float:
            The value with hich to initialise the model biases.
        '''

        # Initialise model parameters
        self.user_latents = None
        self.item_latents = None
        self.user_bias = None
        self.item_bias = None
        self.mu = 0

        # Initialise model hyperparameters
        self.lmbda = self._parse_hparams(lmbda)
        self.learning_rate = None
        self.n_factors = n_factors

        # Initialise other attributes
        self.latent_init = latent_init
        self.bias_init = bias_init
        self.cost = np.inf

    def get_params(self) -> dict:

        '''
        Description:
        Gets the parameters for the current state of the model.

        RETURNS:
        A dictionary with the following keys,
            user_latents -> The values for the user latents.
            item_latents -> The values for the item latents.
            user_bias -> The values for the user biases.
            item_bias -> The values for the item biases.
            mu -> The mean overall rating.
        '''

        return {
            "user_latents":self.user_latents,
            "item_latents":self.item_latents,
            "user_bias":self.user_bias,
            "item_bias":self.item_bias,
            "mu":self.mu
        }

    def fit(self, observed_ratings, learning_rate = 0.1, n_iter = 100) -> None:

        '''
        DESCRIPTION:
        fits the SVD recommender using stochasticc gradient descent. A different
        approach to optimisation was taken compared to (Ricci, F. et al, 2011),
        where only the sign of the gradient was used. Could potentially be
        improved by implementing an ADAM optimiser.

        ARGUMENTS:
        ------------------------------------------------------------------------
        observed_ratings of type np.ndarray [n_users, n_items]:
            A complete matrix of ratings for each item-user pair. No constraints
            are placed on the type of rating used, however it has only been
            tested on binary ratings so far.

        KEYWORD ARGUMENTS:
        ------------------------------------------------------------------------
        learning_rate, of type float:
            The size of the step taken in each iteration of the stochastic
            gradient descent algorithmn, typically should be small and selected
            based on using cross-validation.
        n_iter, of type int:
            The number of stochastic gradient descent algorithmn iterations to
            perform. No convergence criteria has been defined as yet so the
            model will always perform n_iter iterations.

        RETURNS:
        ------------------------------------------------------------------------
        None
        '''

        # Parse learning rate input and update self.learning_rate
        self.learning_rate = self._parse_hparams(learning_rate)

        # Initial update of model parameters
        self.user_latents = np.full((self.n_factors,
                                     observed_ratings.shape[0]),
                                    self.latent_init, dtype = np.float32)
        self.item_latents = np.full((self.n_factors,
                                     observed_ratings.shape[1]),
                                    self.latent_init,  dtype = np.float32)
        self.user_bias = np.full((1, observed_ratings.shape[0]),
                                           self.bias_init, dtype = np.float32)
        self.item_bias = np.full((1, observed_ratings.shape[1]),
                                           self.bias_init, dtype = np.float32)
        self.mu = observed_ratings.values.mean()

        for i in range(n_iter):

            # alias params for code readability
            bu = self.user_bias
            bi = self.item_bias
            uLat = self.user_latents
            iLat = self.item_latents
            λ = self.lmbda
            γ = self.learning_rate

            # Evaluate cost and RMSE at this iteration and print
            pred_error = observed_ratings.values - self.predict()
            self.cost = self._cost(observed_ratings)
            print(("RMSE:{}, "+
                  "Cost:{}").format(np.sqrt((pred_error**2).mean()).round(3),
                                     self.cost.round(2)), end="\r")

            # Adjust parameters
            self.user_bias = (bu + γ[0]*np.sign(pred_error.T - λ[0]*bu))
            self.item_bias = bi + γ[1]*np.sign(pred_error - λ[1]*bi)
            self.user_latents = uLat + (γ[2]*np.sign(iLat.dot(pred_error.T)
                                              - λ[2]*uLat))
            self.item_latents = iLat + (γ[3]*np.sign(uLat.dot(pred_error)
                                              - λ[3]*iLat))

    def predict(self) -> np.ndarray:

        '''
        DESCRIPTION:
        Predicts the utility of each item for each contact based on the current
        fit of the model.

        RETURNS:
        ------------------------------------------------------------------------
        A np.ndarray of shape [n_users X n_items] containing the expected
        utility of each item for each contact
        '''
        # Make predictions
        return (self.mu + self.item_bias.T + self.user_bias
                + (self.item_latents.T.dot(self.user_latents))).T

    def _parse_hparams(self, param_input) -> np.ndarray:

        '''
        DESCRIPTION:
        Takes hyperparameters and parses them to an array. If the recieved
        parameter value is a float, then an array of shape 4, filled with that
        float is returned. If the input is an iterable with exactly four
        elements then that iterable is parsed to an array and returned.

        ARGUMENTS:
        -----------------------------------------------------------------------
        param_input, an iterable of length 4 or a scalar value.
            The values recieved that need to be parsed to a standardised array.

        RETURNS:
        -----------------------------------------------------------------------
        An np.ndarray of shape [4] containing the parsed parameter values.
        '''

        if type(param_input) == float or type(param_input) == int:
            return np.full(4, param_input)
        elif len(param_input) == 4:
            return np.asarray(param_input)
        else:
            raise ValueError("Lambda contained the wrong number of elements."
                             + "For more details call help(SVDRecommender)")

    def _cost(self, observed_ratings) -> float:

        '''

        DESCRIPTION:
        Calculates the cost (sum of squared errors) of the model based on the
        current parameters and a set of observed ratings.

        ARGUMENTS:
        ------------------------------------------------------------------------
        observed_ratings, an np.array of floats with shape [n_users, n_items];
            The observed ratings of each item for each user.

        RETURNS:
        ------------------------------------------------------------------------
        A float giving the value of the cost function for the model based on the
        current parameters.
        '''

        λ = self.lmbda

        predicted_ratings = self.predict()
        squared_errors = (observed_ratings-predicted_ratings)**2
        regularization = (λ[0]*np.linalg.norm(self.user_bias,2)
                          + λ[1]*np.linalg.norm(self.item_bias,2)
                          + λ[2]*np.linalg.norm(self.user_latents, 2)
                          + λ[3]*np.linalg.norm(self.item_latents, 2))
        return squared_errors.values.sum()+regularization
