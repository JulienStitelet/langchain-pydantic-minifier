"""strict test"""

# pylint: disable=wrong-import-position
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain_pydantic_minifier.minifier_pydantic import MinifiedPydanticOutputParser

# will print out the openai payload sent
openai_logger = logging.getLogger("openai._base_client")
openai_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
openai_logger.addHandler(console_handler)

RESUME = """
JOHN MCLANE\
ETUDIANTE EN GOUVERNANCE REGIONALE

Recherche un stage de 4 à 6 mois

au sein d\u0027une ONG humanitaire en Afrique de l\u0027Ouest

22 ans

8 rue des DieHard

75001 Paris

07.99.99.01.00


Linkedin

Permis B

COMPETENCES :\

Maîtrise journalistique :\
Tournage et montage vidéo,
reportage radio, rédaction
d\u0027articles : brève, compte-rendu,
portrait, reportage, interview...
Réalisation d\u0027un blog sur une
immersion au Sénégal .\
  Gestion de projets :\
Organisation et supervision de
projets pour l\u0027Ambassade ex :
mission caritative pendant le
Ramadan, pour une asso
humanitaire ex : collecte de
fournitures scolaires pour des
écoles congolaises... De la
création du dossier de conception
à sa mise en place concrète.\
     Linguistique :\
 Anglais B1 - courant\
Arabe A2 - Lecture arabe littéraire

+ notions orales

Wolof A1 : notions de base\

     Bureautique :\
PackOffice - OpenOffice
EXPERIENCES PROFESSIONNELLES :
Chargée de communication et de relations presse
avril à septembre 2019
Ambassade de France, Dakar, Sénégal
Rédaction de notes diplomatiques sur la société sénégalaise pour le Quai
d\u0027Orsay, rédaction de dossiers de presse, gestion et coordination des
projets événementiels de l\u0027Ambassade, organisation des interviews de
l\u0027Ambassadeur, gestion des réseaux sociaux...
Stagiaire journaliste au service Afrique - février à avril 2019
RFI, Issy-les-Moulineaux
Préparation de l\u0027émission « Alors on dit quoi » : recherche et interviews
des invités, rédaction du script, coordination des sujets des
correspondants en Afrique...
Stagiaire journaliste au service Culture - avril à juillet 2017
L\u0027Orient-Le Jour - Beyrouth, Liban
Couverture de concerts, expositions, conférences de presses. Rédaction
de portraits, interviews, reportages...
FORMATION :
Master en Gouvernance régionale - Afrique - Depuis 2019
Sciences Po - Grenoble
Cours : plaidoyer, analyse géopolitique, droit en Afrique de l\u0027Ouest,
migrations africaines, environnement et développement en Afrique...
Bachelor en journalisme Mention Très bien - 2016/2019
Institut Supérieur de Médias - Lyon
Baccalauréat Littéraire Mention Bien - 2015
Lycée Mathias - Chalon sur Saône
CENTRES D\u0027INTERET :
Voyages :
Allemagne, Angleterre, Tunisie, Liban, Maroc, Sénégal,
préparation d\u0027un voyage humanitaire au Congo.
Sports :
Athlétisme et basket-ball en club pendant 6 ans
Milieu associatif :
Bénévole au Secours Populaire depuis 4 ans et bénévole
chez « Protect Talibés » (Sénégal) depuis 3 ans."
"""


class ProfileResponse(BaseModel):
    """
    All informations relative to the owner,candidate, of the resume.
    """

    prof_name: Optional[str] = Field(
        default=None, description="String containing first name of the candidate."
    )
    prof_lastname: Optional[str] = Field(
        default=None, description="String containing last name of the candidate."
    )
    prof_date_of_birth: Optional[str] = Field(
        default=None,
        description="String containing date of birth of the candidate. Format it like YYYY-MM-DD.If no date is available, leave it blank.",
    )
    prof_gender: Optional[str] = Field(
        default=None, description="String containing gender of the candidate."
    )
    prof_email: Optional[str] = Field(
        default=None, description="String containing email of the candidate."
    )
    prof_home_phone: Optional[str] = Field(
        default=None, description="String containing home phone of the candidate."
    )
    prof_mobile_phone: Optional[str] = Field(
        default=None, description="String containing mobile phone of the candidate."
    )
    prof_country_code: Optional[str] = Field(
        default=None,
        description="String containing the country of the candidate. Return the standard ISO country code for the given country name (e.g., 'United States' as 'US', 'Canada' as 'CA').",
    )
    prof_city: Optional[str] = Field(
        default=None, description="String containing city of the candidate."
    )
    prof_state: Optional[str] = Field(
        default=None, description="String containing region or state of the candidate."
    )
    prof_address: Optional[str] = Field(
        default=None, description="String containing address of the candidate."
    )
    prof_postcode: Optional[str] = Field(
        default=None, description="String containing postal code of the candidate."
    )
    prof_has_managed_others: Optional[bool] = Field(
        default=None,
        description="boolean indicating if the candidate has managed others.",
    )
    prof_drivers_license: Optional[list[Optional[str]]] = Field(
        default=[], description="Driver licenses of the candidate."
    )
    prof_social_media_links: Optional[list[Optional[str]]] = Field(
        default=[],
        description="Social media http links of the candidate.Format must be HTTP link.",
    )
    prof_salary: Optional[str] = Field(
        default=None,
        description="String containing salary information of the candidate",
    )


model = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=10,
    temperature=0.0,
    max_retries=0,
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You'll be given a raw text resume. Your role is to extract the profile information from it. Use the structured output schema provided in the query."
        ),
        MessagesPlaceholder(variable_name="history_list"),
        MessagesPlaceholder(variable_name="human_message", optional=True),
    ]
)

# Set up a parser
parser = MinifiedPydanticOutputParser(pydantic_object=ProfileResponse, strict=True)

# chain = chat_prompt | model.with_structured_output(parser.minified, strict=True)
chain = chat_prompt.pipe(model.with_structured_output(parser.minified, strict=True))

llm_response = chain.invoke(
    {
        "history_list": [],
        "human_message": [HumanMessage(content=RESUME)],
    }
)

r = parser.get_original(llm_response)
# Print the response
print(llm_response)
print(r)
