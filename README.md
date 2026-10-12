# Customer Segmentation Using Hierarchical Clustering

Minor project (AI/ML): groups mall customers by Annual Income and Spending Score using Agglomerative (hierarchical) clustering.

## Files
- `hierarchical_customer_clustering.py` - main script
- `hierarchical_customer_clustering.ipynb` - same code as a notebook
- `app.py` - Streamlit app to predict a customer's segment
- `build_notebook.py` - regenerates the notebook from the script
- `Mall_Customers.csv` - dataset (download from Kaggle, put it here)
- `outputs/` - generated graphs and CSVs

## How to run
```
pip install -r requirements.txt
python hierarchical_customer_clustering.py
streamlit run app.py
```
