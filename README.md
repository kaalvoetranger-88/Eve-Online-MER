# EVE Online Monthly Economic Report (MER) Repository

Welcome to the EVE Online Monthly Economic Report (MER) GitHub repository! This repository is dedicated to collecting and analyzing the Monthly Economic Reports published by CCP Games for EVE Online's in-game economy.

## About EVE Online's MER
EVE Online's Monthly Economic Report provides valuable insights into the economic activities and trends within the EVE universe. The MER includes data on various aspects of the game economy, such as market activity, mining output, trade volumes, and more.
The MER is published on the Eve Online website in the first week of every month, with the datasets used to build the report being easily downloadable:

[Eve Online Website](https://www.eveonline.com/news/search?q=monthly%20economic%20report)

### Purpose of this Repository
The goal of this repository is to:

- Archive historical Monthly Economic Reports for easy access and reference.
- Facilitate analysis and visualization of EVE Online's economic data.
- Encourage collaboration and contribution from the EVE Online community in analyzing the MER data.

## Repository Structure

~~~
├── data
│   ├── STATIC_ore_type_mapping.csv
│   ├── STATIC_triglavian_sites.csv
│   ├── commodity_sinks_and_faucets_history.csv
│   ├── economy_indices_details.csv
│   ├── index_baskets.csv
│   ├── mining_by_region.csv
│   ├── mining_history_by_security_band.csv
│   ├── money_supply.csv
│   ├── produced_destroyed_mined.csv
│   ├── regional_stats.csv
│   ├── sinks_and_faucets_history.csv
│   ├── wormhole-trade.csv
│   ├── STATIC_ore_type_mapping.csv
│   └── STATIC_ore_type_mapping.csv
├── analysis
│   ├── scripts
│   │   └── analyze_mer.py
│   └── visualizations
│       └── mer_visualizations.ipynb
├── .gitignore
├── LICENSE
└── README.md
~~~

- data/: Contains most recent Monthly Economic Report data. There are 14 CSV files and 23 HTML files. CCP standardised this data stream in 2019. 
- analysis/: Includes scripts and notebooks for analyzing and visualizing MER data.
- .gitignore: Specifies files and directories to ignore in version control.
-  LICENSE: Describes the terms of use for the data and code in this repository.

## Getting Started
### Accessing Monthly Economic Report Data
You can find the most recent monthly report data in the data/ directory. The data is stored in HTML and CSV formats.

### Analyzing the Data
To analyze the MER data, you can use the provided scripts and tools in the analysis/ directory. Refer to the README files within each directory for more information on how to use them.

## Contributing
Contributions to this repository are welcome! If you have scripts, tools, or analyses related to EVE Online's economic data, feel free to submit a pull request. Please follow these guidelines:

- Fork the repository and create a new branch for your changes.
- Ensure your code is well-documented and adheres to best practices.
- Submit a pull request describing your changes and their purpose.

## License
This repository is licensed under the [MIT License](https://opensource.org/license/mit). You are free to use, modify, and distribute the code and data within this repository, provided you include the appropriate attribution.

##Feedback and Suggestions
If you have any feedback, suggestions, or questions regarding this repository, please feel free to open an issue or contact the [maintainer](mailto:kaalvoetranger@gmail.com?subject=MER_github).


