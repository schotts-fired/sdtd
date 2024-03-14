import requests


def download_dataset(url: str, save_path: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Failed to download dataset. Status code:", response.status_code)


class Domain:
    """A class to represent the domain of a feature. Allows defining and checking for arbitrary properties of the domains, such as positivity, continuity, etc..

    Args:
        name (str): The name of the domain as specified by the user. One of ['binary', 'positive', 'real', 'discrete'].
    """

    SUPPORTED_DOMAINS = ['binary', 'positive', 'real', 'discrete']

    def __init__(self, name: str):
        assert name in self.SUPPORTED_DOMAINS, f"Domain {name} not supported. Supported domains: {self.SUPPORTED_DOMAINS}"
        self.name = name

    def is_binary(self) -> bool:
        return self.name == 'binary'

    def is_positive_real(self) -> bool:
        # positive is a subset of continuous here
        return self.name == 'positive'

    def is_real(self) -> bool:
        return self.name == 'real'

    def is_continuous(self) -> bool:
        return self.name == 'real' or self.name == 'positive'

    def is_discrete(self) -> bool:
        return self.name == 'discrete'

    def __repr__(self) -> str:
        return f"Domain({self.name})"

    def __str__(self) -> str:
        return self.name
