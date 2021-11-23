
# D3TECT implementation
This is the public repo of D3TECT. 

Paper results can be found in the [work directory](work/) as 
* [Jupyter Notebook](https://github.com/d3tect/d3tect/blob/main/work/D3TECT%20Paper%20Input.ipynb) 
* and as an HTML export.
For best results please download the HTML file or the Jupyter Notebook and run it locally as Github restricts the use of Javascript.

[Stix data exports](stix-data) are [extracted](d3tect/extract-attack-stix.py) from [MITRE's ATT&CK Framework](https://attack.mitre.org/) published in [stix format](https://github.com/mitre-attack/attack-stix-data).

Threat-actor-data yaml files of rabobank's DeTTECT were used for the evaluation of some of the metrics. If you want to re-compile the Jupyter Notebook some parts require the download of the tool to D3TECT's root directory.

```
git clone git@github.com:d3tect/d3tect.git
cd d3tect
git clone --depth 1 --branch v1.4.4 https://github.com/rabobank-cdc/DeTTECT.git 
```
