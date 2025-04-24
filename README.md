# Added_Marginal_Wasserstein_Distance
Contains code necessary to replicate results from "Additive Marginal Wasserstein Distances as a Computationally Efficient Multidimensional Distance Measure", a poster presentation at the West Virginia University Spring 2025 Undergraduate Research Symposium. 

# Explanation
Added Marginal Wasserstein Distance (AMWD) is an alternative to calculating the full Wasserstein distance in cases with more than 1 dimension. The goal of the project was to measure the correspondence between AMWD and the n-dimensional Wasserstein distance, where a higher degree of correspondence would be an indicator of AMWD's validity as an alternative. In essence, AMWD is viable if it is high when n-dim Wasserstein distance is high and low when n-dimensional Wasserstein distance is low (in terms of distributional shape; the two measures vary widely in their scale).

The motivation for an alternative is the fact that n-dim Wasserstein distance can be computationally expensive with a high n, but is very fast when n is 1. AMWD leverages this point by simply taking the sum of the 1-dim Wasserstein distances for each feature and using that as a measure of the distance between the two n-dim distributions. While a computationally cheap alternative already exists in Sliced Wasserstein distance and its many variations, but it is not very easily interpretable due to the nature of the projections of the variables into a one-dimensional space and the averaging. AMWD introduces an alternative to this problem which, although it unforunately does scale with the number of dimensions, is very easily interpretable: an AMWD value is the sum the Wasserstein distances between the distributions of each feature. 

# Running Code
To run the code contained in this repo, you will likely need to download the Python Optimal Transport (POT) library (https://github.com/PythonOT/POT) if you do not already have it downloaded. The full citation is included below. 

Python Optimal Transport (POT) Library GitHub and Math:
Flamary R., Vincent-Cuaz C., Courty N., Gramfort A., Kachaiev O., Quang Tran H., David L., Bonet C., Cassereau N., Gnassounou T., Tanguy E., Delon J., Collas A., Mazelet S., Chapel L., Kerdoncuff T., Yu X., Feickert M., Krzakala P., Liu T., Fernandes Montesuma E. POT Python Optimal Transport (version 0.9.5). URL: https://github.com/PythonOT/POT

Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer, POT Python Optimal Transport library, Journal of Machine Learning Research, 22(78):1−8, 2021. URL: https://pythonot.github.io/
