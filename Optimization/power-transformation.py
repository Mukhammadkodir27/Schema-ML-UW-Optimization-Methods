# lets try to apply Box-Cox transformation for Sale_Price
#  (Sale_Price must be > 0)

# Apply boxcox and get the optimal lambda
# (optimized if you do not provide the lmbda= argument)
sale_price_boxcox, fitted_lambda = boxcox(houses_train_encoded['Sale_Price'])

print(fitted_lambda)


# Add transformed column to dataframe
houses_train_encoded['Sale_Price_boxcox'] = sale_price_boxcox

# !!! If lambda is close to 0, you may use log transformation instead

# To invert the Box-Cox transformation (to go back to original values), use:
# from scipy.special import inv_boxcox
# inv_boxcox(transformed_values, fitted_lambda)

# check the first few rows of the transformed column
print(houses_train_encoded['Sale_Price_boxcox'].head())
