"""test parser"""

# pylint: disable=wrong-import-position
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
import time
from typing import Optional

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain_pydantic_minifier.minifier_pydantic import MinifiedPydanticOutputParser

# will print out the openai payload sent
openai_logger = logging.getLogger("openai._base_client")
openai_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
openai_logger.addHandler(console_handler)

# SAMPLE JOURNALIST RESUME
# *This is a fictional example for template/educational purposes only*

RESUME = """
## PROFILE
**Name:** Alexandra Martinez  
**Address:** 123 Sample Street, Example City, ST 12345  
**Phone:** (555) 123-4567  
**Email:** a.martinez@email.example  
**LinkedIn:** linkedin.com/in/example-profile  
**Portfolio:** www.example-portfolio.com  

---

## EDUCATION

**Master of Arts in Journalism**  
Columbia University Graduate School of Journalism | New York, NY  
*September 2018 - May 2020*
- Concentration in Investigative Reporting
- GPA: 3.8/4.0
- Thesis: "Digital Media's Impact on Local News Coverage"

**Bachelor of Arts in English Literature**  
University of California, Berkeley | Berkeley, CA  
*September 2014 - May 2018*
- Minor in Political Science
- Magna Cum Laude, GPA: 3.7/4.0
- Editor-in-Chief, The Daily Californian (2017-2018)

---

## PROFESSIONAL EXPERIENCE

**Senior Staff Writer**  
*The Metropolitan Herald* | New York, NY  
*June 2021 - Present*
- Cover breaking news and conduct in-depth investigative pieces on local government
- Maintain relationships with key sources across city departments
- Average 3-4 articles per week with readership of 50,000+ per piece
- Led coverage of 2022 mayoral election, resulting in 25% increase in digital engagement

**Staff Reporter**  
*Bay Area News Network* | San Francisco, CA  
*August 2020 - May 2021*
- Reported on technology sector, startups, and business developments
- Conducted over 200 interviews with industry executives and entrepreneurs
- Broke three major stories resulting in federal investigations
- Managed social media presence, growing followers by 150%

**Freelance Journalist**  
*Various Publications* | Remote  
*May 2018 - August 2020*
- Published articles in The Washington Post, Wired, and The Atlantic
- Specialized in technology ethics and privacy issues
- Developed expertise in data journalism and statistical analysis
- Maintained 98% on-time delivery rate for commissioned pieces

**Editorial Intern**  
*San Francisco Chronicle* | San Francisco, CA  
*June 2017 - August 2017*
- Assisted senior reporters with research and fact-checking
- Wrote 15 published pieces on local community events
- Transcribed interviews and maintained editorial calendar
- Gained experience with content management systems

---

## SKILLS

**Technical Skills:**
- Content Management Systems (WordPress, Drupal)
- Social Media Management (Twitter, Facebook, Instagram, LinkedIn)
- Data Analysis (Excel, Google Analytics, basic SQL)
- Video Editing (Adobe Premiere Pro, Final Cut Pro)
- Photography (Adobe Photoshop, Lightroom)
- Audio Editing (Audacity, Adobe Audition)

**Journalism Skills:**
- Investigative Reporting
- Breaking News Coverage
- Interview Techniques
- Fact-Checking and Verification
- FOIA Request Processing
- Court Reporting
- Beat Reporting (Government, Technology, Business)

**Languages:**
- English (Native)
- Spanish (Fluent)
- French (Conversational)

**Software Proficiency:**
- Microsoft Office Suite
- Google Workspace
- Slack, Zoom
- AP Stylebook
- Transcription software

---

## AWARDS & RECOGNITION

- Regional Press Association Award for Investigative Reporting (2023)
- Society of Professional Journalists Excellence in Journalism Award (2022)
- University Alumni Association Outstanding Graduate Award (2020)

---

## PROFESSIONAL MEMBERSHIPS

- Society of Professional Journalists
- National Association of Hispanic Journalists
- Investigative Reporters and Editors (IRE)

---

*References available upon request*
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

default_parser = PydanticOutputParser(pydantic_object=ProfileResponse)
minifying_parser = MinifiedPydanticOutputParser(pydantic_object=ProfileResponse)
results = {"default_parser": {}, "minifying_parser": {}}
query = RESUME

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract information from the given raw text resume. 
            Wrap the output in `json` tags\n{format_instructions}""",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=default_parser.get_format_instructions())
# print(default_prompt.invoke({"query": query}).to_string())
chain = default_prompt.pipe(model).pipe(default_parser)
start = time.time()
with get_openai_callback() as cb:
    llm_response = chain.invoke({"query": query})
    results["default_parser"]["time"] = time.time() - start
    results["default_parser"]["cb"] = cb


minifying_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract information from the given raw text resume. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=minifying_parser.get_format_instructions())
# print(minifying_prompt.invoke({"query": query}).to_string())
chain = minifying_prompt.pipe(model).pipe(minifying_parser)

start = time.time()
with get_openai_callback() as cb:
    llm_response = chain.invoke({"query": query})
    results["minifying_parser"]["time"] = time.time() - start
    results["minifying_parser"]["cb"] = cb


# Print the response
print(results)
print(llm_response)
