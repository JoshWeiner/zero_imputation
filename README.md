# zero_imputation
Repository for zero imputation repo, Published ICML 2024

### Abstract
Zero-imputation methods are widely applied to address non-biological zeros in scRNA-seq data. However, these methods can introduce artificial signals, skewing the results of downstream analysis to match initial assumptions rather than emulate the underlying biological processes. This paper makes a simple but surprising observation: we demonstrate that several popular zero imputation techniques provide significantly varied results on the downstream network inference tasks over the same real-world scRNA datasets. Benchmarking their performance on synthetically controlled simulated scRNA datasets using the SERGIO simulator and the GENIE3 network inference algorithm, we observed poor metrics across the board. A key takeaway from our analysis is both unearthing the unreliability of existing imputation techniques and the inability to define a uniform gold-standard for zero imputation.

### Link to Paper
[ICML Publication](https://openreview.net/pdf?id=VAOlHfzIUh)
