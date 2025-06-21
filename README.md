# NumpyGCRL: A Offline Goal-Conditioned Reinforcement Learning Framework

NumpyGCRL is a lightweight and extensible framework for Offline Goal-Conditioned Reinforcement Learning (Offline-GCRL), inspired by repositories such as OGBench, TD3BC, and DiffusionQL. This repository aims to provide a foundation for implementing and reproducing offline reinforcement learning algorithms in a goal-conditioned setting.

## Features
- **Algorithm Support**: Includes implementations of algorithms like GCBC, GCTD3BC, GCDiffusionQL, and more.
- **Dataset Handling**: Provides utilities for working with datasets, including hierarchical goal-conditioned datasets.
- **Environment Integration**: Seamlessly integrates with D4RL and OGBench environments.
- **Extensibility**: Designed to be modular, allowing easy addition of new algorithms, environments, and datasets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NumpyGCRL.git
   cd NumpyGCRL
2. Install dependencies: 
    ```bash
    pip install -r requirements.txt
## Getting Started
1. To train a GCBC agent:
    ```bash
    python main.py --algo GCBC
## Repository Structure
NumpyGCRL/

├── algos/                # Algorithm implementations

├── models/               # Model definitions (e.g., policies, Q-functions)

├── utils/                # Utility functions and dataset handling

├── [main.py](http://_vscodecontentref_/1)               # Entry point for training and evaluation

├── [README.md](http://_vscodecontentref_/2)             # Project documentation

└── requirements.txt      # Python dependencies
## Contributing
We welcome contributions from the community to improve this repository and expand its capabilities. Here are some ways you can contribute:
   1. Add New Algorithms: Implement and integrate additional offline RL algorithms.
   2. Improve Documentation: Help us enhance the documentation for better usability.
   3. Bug Fixes: Report and fix issues in the existing codebase.
   4. Benchmarking: Provide benchmarks for existing algorithms on new datasets or environments.

To contribute, please fork the repository, create a new branch, and submit a pull request. For major changes, consider opening an issue first to discuss your ideas.
## Acknowledgments
This repository is inspired by and builds upon the ideas from:
1. [OGBench](https://github.com/seohongpark/ogbench)
2. [TD3BC](https://github.com/sfujim/TD3_BC)
3. [DiffusionQL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
   
Be sure to reference them if you need them.
## License
This README is designed to encourage community contributions while providing clear instructions for usage and development. You can customize it further based on your specific goals and requirements.
