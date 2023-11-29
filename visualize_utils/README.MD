### Check file results:
- Clustering by Kmean
- Considering 3 attributes: "direction" (straight, reverse, ..), "p" (20, 30, ..), and "area_type" (even or odd)
- Split features by "features size" and "start_pos" (index of  features from "start" -> "start + features size")
- Considering 3 evaluation metrics ([docs](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)): "V-measure", "Rand index", and "NMI".  
- Run `python main.py` with appropriate `path_data` variable. 