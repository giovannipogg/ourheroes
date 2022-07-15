"""Factory methods for the `digesting` package.

The default digester retains the four most relevant
sections for each of these only the top 30 relevant
sentences as in the reference paper.
"""

from ourheroes.training.digesting.digester import Digester


def default_digester() -> Digester:
    """Creates the default Digester object.

    Returns:
        The default Digester object.
    """
    return Digester(4, 30, 'sections')
