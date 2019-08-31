Release 2.1
25th March 2019

This directory contains the official version of the First Certificate in English (FCE) corpus used in the BEA2019 shared task.

More details about the FCE corpus can be found in the following paper:

Helen Yannakoudakis, Ted Briscoe, and Ben Medlock. 2011. A new dataset and method for automatically grading ESOL texts. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 180–189.

The original FCE files are available here: https://ilexir.co.uk/datasets/index.html
The raw dataset is not explicitly split into training, development and test sets, and so we recreated this split based on the error detection version of the dataset available at the same link.

This version of the public FCE is available in two different formats: JSON and M2.

-- JSON --
The JSON format is the raw unprocessed version of the corpus. Each line in a JSON file contains the following fields:
    id       : A unique id for the essay.
    l1       : The first language of the author.
	age      : The age (or age range) of the author.
	q        : The question number; each author submitted essay answers to 2 different questions.
	answer-s : The score awarded to the essay for this particular question.
	script-s : The overall score awarded to the author for both questions they answered.
    text     : The essay as it was originally written by the author.
    edits    : A list of all the character level edits made to the text by all annotators, of the form:
               [[annotator_id, [[char_start_offset, char_end_offset, correction], ...]], ...].

-- M2 --
The M2 format is the processed version of the corpus that we recommend for the BEA2019 shared task.
M2 format has been the standard format for annotated GEC files since the first CoNLL shared task in 2013.

Since it is not easy to convert character level edits in unprocessed text into token level edits in sentences (cf. https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-894.pdf), we provide a json_to_m2.py script to convert the raw JSON to M2. This script must be placed inside the main directory of the ERRor ANnotation Toolkit (ERRANT) in order to be used. ERRANT is available here: https://github.com/chrisjbryant/errant

Each M2 file was thus generated in Python 3.5 using the following command:

python3 errant/json_to_m2.py <wi_json> -out <wi_m2> -gold

This used spacy v1.9.0 and the en_core_web_sm-1.2.0 model.

Updates
----------------------

-- v2.0 --

* Added new JSON files for the FCE if users want the original data in the same format as the W&I+LOCNESS corpus.

* All punctuation was normalised in the M2 files. It was otherwise arbitrary whether, for example, different apostrophe styles were corrected or not.

* Fixed a bug in the character to token edit conversion script.

* Fixed a bug with correction edits nested inside detection edits that led to them being ignored.

-- v2.1 --

* Updated the json_to_m2.py script to handle multiple annotators.
