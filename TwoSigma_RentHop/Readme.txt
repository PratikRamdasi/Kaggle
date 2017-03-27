Overview:
=========
Finding the perfect place to call your new home should be more than browsing through endless listings.RentHop makes apartment 
search smarter by using data to sort rental listings by quality. But while looking for the perfect apartment is difficult 
enough, structuring and making sense of all available real estate data programmatically is even harder. 

Objective:
==========
To predict how popular an apartment rental listing is based on the listing content like text description, photos, 
number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website.These apartments are located 
in New York City. The target variable, interest_level, is defined by the number of inquiries a listing has in the duration 
that the listing was live on the site.

Models Developed:
=================
1. Random Forest classifier
2. Xtereme Gradient Boosting (XGBoost)
3. Principle Component Analysis for dimension reduction with SVM classifier

Evaluation:
==========
Models are evaluated using the multi-class logarithmic loss. Random Forest and XGBoost give similar perforance with multi-class 
loss around 0.62.  PCA with SVM does not found to be suitable with loss around 0.90. As the dataset is huge with many features 
(Preprocessed 11 features), RF and XGBoost outperform PCA + SVM. 
