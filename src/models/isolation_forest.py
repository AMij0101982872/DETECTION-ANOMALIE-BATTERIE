"""Modèle 3 — Atypique (non supervisé, Isolation Forest) : « cette charge est bizarre ».

À FAIRE (membre 4) :
- entrée : toutes les features de session, normalisées, sans réponse
- sortie : un score de bizarrerie par session
- évaluation (§3.7) : injecter des surchauffes et des décrochages de cellule simulés dans
  des sessions normales et vérifier que le modèle les retrouve
"""


def entrainer(sessions):
    raise NotImplementedError


def vote_sessions(sessions, modele):
    """1 si la session est classée atypique, sinon 0."""
    raise NotImplementedError
