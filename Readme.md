# Quantitative Image Analysis using Python
## Main purpose: Analysing HEK Cell transfections

This code will analyse flourescence microscopy images of cells transfected with flourescently tagged proteins with different localizations.
It will output a classic boxplot with scatter for each datapoint and calculate statistical significance for each condition *via* t-test if the data is normally distributed.

#### By default, the channels are expected to be in the following order:
1. Brightfield image
2. Signal localized in the cytoplasm
3. Signal localized in the nucleus

### Important!
The program assumes the data is in a specific folder and different channels are present as their own unique files.
All files should be named starting with an increasing number for each condition.

**Folder, file extentions and sample names have to be enterted inside the code!**, below the `if __name__ == "__main__"`
