# Datasheet for dataset "PsyCoMark - Psycholinguistic Conspiracy Marker Dataset"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

### For what purpose was the dataset created? 

The dataset was created to address the lack of benchmarks for detecting conspiracy-related content in everyday online conversational settings. It introduces a novel dataset and span identification task focused on uncovering psycholinguistic markers of conspiracy theories. The goal is to improve the performance of conspiracy detection models by leveraging these markers in topic-agnostic conversations.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

The dataset was created by Mattia Samory (Sapienza University of Rome), Felix Soldner (GESIS Cologne), and Veronika Batzdorfer (Karlsruhe Institute of Technology).

### Who funded the creation of the dataset? 

The dataset creation was supported, in part, by the Sapienza grant 000090_22_SEED_PNR-_SAMORY and by GESIS.

### Any other comments?

The dataset aims to advance the computational analysis of conspiracy theory language by focusing on psychologically grounded markers and topic-agnostic conversations, addressing limitations in existing corpora.

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

The instances in this dataset are submission statements from the Reddit platform. These are first-level comments authored by the discussion starter, typically accompanying media submissions, and summarizing the media's relation to the subreddit's topic.

### How many instances are there in total (of each type, if appropriate)?
The training split of the dataset presently consists of 4361 submission statements.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset is a sample of submission statements from Reddit. The sampling strategy involved selecting comments based on specific criteria, including starting with "ss|((submission )?statement)[^a-zA-Z\d]:" and falling within a character length of 160 to 1000. Comments containing quotes were discarded to focus on the author's original text. To address the relative rarity of conspiracy theories on Reddit, the dataset oversampled comments likely to contain them, including 1150 comments from the subreddit r/conspiracy and the remaining comments from other subreddits with at least 10 submission statements. This oversampling means the dataset is not fully representative of the overall distribution of submission statements on Reddit. 

### What data does each instance consist of? 

Each instance consists of one annotated submission statement. Each data point includes:
- `_id` (str): the Reddit fullname of the submission statement
- `conspiracy` (str): one of `Yes`, `No`, `Can't tell`
- `markers` (list of dict): list of psycholinguistic marker spans extracted from the submission statement; each is structured as follows:
	- `startIndex` (int): first character in the preprocessed text
	- `endIndex` (int): last character in the preprocessed text
	- `type` (str): one of `Actor`, `Action`, `Victim`, `Threat`, `Evidence`
	- `text` (str): the plain text of the marker
- `subreddit` (str): the subreddit of the submission statement	
- `annotator` (str): the pseudonymized id of the annotator


### Is there a label or target associated with each instance?

Yes, each instance is associated with two types of labels:
- Conspiracy Detection: Each comment is classified as either "conspiracy," "not conspiracy," or "can't tell".
- Conspiracy Marker Extraction: Each comment may contain zero or multiple spans annotated for five specific psycholinguistic markers: Actor, Effect, Victim, Evidence, and Action.

### Is any information missing from individual instances?

Raw text of the submission statements is not released, and should be rehydrated from Reddit archives. Preprocessing steps on the raw text should also be replicated to match marker positions. Preprocessing includes removal of URLs and stripping leading and trailing whitespace. Rehydration and preprocessing can be accomplished through the script provided at this link: https://github.com/hide-ous/semeval26_starter_pack

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

The primary relationship made explicit is the annotation of conspiracy labels and psycholinguistic markers within each individual submission statement. The dataset also retains information about the subreddit from which each comment was sourced, which implicitly links comments within the same subreddit.

### Are there recommended data splits (e.g., training, development/validation, testing)?

The data shared is a random sample of ~90% of the annotated data, intended for model training and development. The test set was retained as private to enable participation in competitive shared tasks.

### Are there any errors, sources of noise, or redundancies in the dataset?

The inter-annotator agreement for the binary conspiracy labels is measured as Krippendorff's α=0.58. This indicates a moderate level of agreement, suggesting some inherent ambiguity or noise in the labeling process. The "can't tell" category also acknowledges instances where classification was difficult (α=0.50 if including the "can't tell" label). A portion of the data was annotated by multiple annotators and the training set does include repeated entries. The plain text of the submission statements may become unavailable as authors and moderators may remove them.


### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

Rehydration of raw text relies on either Reddit's API or third-party Reddit archives. The rehydration script linked in this document relies on Project Arctic Shift. The continued availability of the original Reddit content is not guaranteed.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

The raw data was obtained from publicly accessible archives on Reddit. The dataset is released without author identifiers and raw comment text to preserve privacy.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

Given the topic of conspiracy theories and the diverse range of subreddits from which the data was collected (including subreddits like conspiracy, ufobelievers, epstein, hongkong, truecrime, news), it is possible that some content within the dataset could be perceived as offensive, insulting, threatening, or anxiety-inducing by some individuals. Conspiracy theories can touch upon sensitive topics and may contain strong opinions or unsubstantiated claims.

### Does the dataset relate to people? 

Yes, the dataset contains text written by people in the form of Reddit comments.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

The provided text does not indicate that specific demographic subpopulations like age or gender are explicitly identified or annotated within the dataset. The data is categorized by conspiracy-relatedness and the presence of psycholinguistic markers.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

The data is released without personal identifiers and raw comment text to preserve privacy. However, the original Reddit comment, which might contain usernames or other potentially identifying information, could be retrieved.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

Reddit comments can potentially contain a wide range of personal opinions and information, including political opinions, religious beliefs, and potentially other sensitive topics depending on the subreddit and the content of the discussion. The dataset's includes comments from subreddits related to politics, news, and specific events which may lead to encountering such sensitive information.

### Any other comments?

## Collection process

### How was the data associated with each instance acquired?

The data consists of submission statements selected from Reddit archives. The selection process involved identifying comments based on specific textual patterns (starting with "ss" or "submission statement") and length. Subreddits who used submission statements for purposes other than explaining event- or narrative- based media in the original submission were discarded.
The subreddits that were manually excluded are the following: 'u_PlanetToday', 'supremeclothing', 'sneakermarket', 'rccars', 'portugueses', 'outlinevpn', 'microgrowery', 'marioandluigi', 'libertarianmeme', 'leagueoflegends', 'juggling', 'fo4', 'education', 'baseball', 'VaporwaveArt', 'SubSimGPT2Interactive', 'Plumbing', 'PhotoshopRequest', 'NBASpurs', 'NBASpurs', 'HollywoodUndead', 'Greyhawk', 'Filmmakers', 'FightingFakeNews', 'Fallout', 'EDC', 'CplSyx', 'CplSyx', 'Aquariums', 'juggling', 'Degrassi', 'coasttocoastam', 'coasttocoastpm', 'coasttocoast', 'woodworking','whatisthisthing' ,'vinyl', 'vexillology' ,'pcmasterrace', 'movies', 'dataisbeautiful', 'cinematography',  'cats', 'backpacking', 'adventures','Homesteading','EuroArchitecture', 'AfricanArchitecture',  'AccidentalRenaissance', 'Greyhawk', 'DronedOrc','DnD', 'CombatFootage','Boruto', 'AskReddit', 'Aquariums','pics', 'Watches'

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

The data was collected using a python script matching regular expressions in comments from the Pushshift and Arctic Shift Project Reddit archives. 

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

The dataset is a sample of Reddit submission statements. The sampling strategy involved a combination of uniform random sampling from the conspiracy subreddit and stratified sampling from other subreddits with at least 10 submission statements after preprocessing.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
The annotation process was conducted through the crowdwork platform Prolific, with annotators being compensated at an average rate above £9/h.

### Over what timeframe was the data collected?

Data was collected between January and March 2025. The timespan of the instances is March 2013 to December 2023.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

Yes, the paper mentions obtaining IRB approval from GESIS Cologne for the annotation task, although the raw data was obtained from publicly accessible archives.

### Does the dataset relate to people?

Yes, the dataset relates to people as it refers to their written comments.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
The data was obtained from third-party archives Pushshift and Arctic Shift Project.

### Were the individuals in question notified about the data collection?

Individual users were not directly notified about the collection.

### Did the individuals in question consent to the collection and use of their data?

Consent was not directly obtained from individual Reddit users for the collection of publicly available comments.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

Users can request exclusion from the data sources used.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

We release the dataset without author identifiers and raw comment text to preserve privacy, in consideration of the potential impact on data subjects. The aim of the research is to understand and detect conspiracy theories, which could indirectly impact individuals who hold or are affected by such beliefs. 

### Any other comments?

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

Yes, preprocessing steps were performed:
- Submission statements were selected based on starting with the specified phrase and having a length between 160 and 1000 characters.
- Markdown formatting within the comments was converted to plain text.
- URLs present in the comments were converted to special tokens `[URL]`.
- Comments containing quotes were discarded to limit the text to the comment's author.
- The remaining comments were then annotated for conspiracy labels ("conspiracy," "not conspiracy," "can't tell") and the five psycholinguistic markers (Actor, Effect, Victim, Evidence, Action).

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The original Reddit identifiers are retained, allowing for potential retrieval of the raw data, subject to Reddit's availability.

### Is the software used to preprocess/clean/label the instances available?

Preprocessing can be accomplished through the script provided at this link: https://github.com/hide-ous/semeval26_starter_pack

### Any other comments?
The annotation process involved training annotators on the task and codebook, and implementing quality control measures such as discarding entire batches if an annotator did not annotate more than two markers in at least one document.

## Uses

### Has the dataset been used for any tasks already?

No.

### Is there a repository that links to any or all papers or systems that use the dataset?

No.

### What (other) tasks could the dataset be used for?
This dataset could be used for various research tasks, including:
- Developing and evaluating more advanced models for conspiracy marker extraction.
- Improving the accuracy of conspiracy detection in online conversations.
- Investigating the relationship between psycholinguistic markers and the expression of conspiracy beliefs.
- Analyzing the linguistic differences between conspiracy-related and non-conspiracy-related discourse across different topics.
- Studying the prevalence and patterns of conspiracy thinking in different online communities (subreddits).
- Exploring the explainability of conspiracy detection models by leveraging the identified markers.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Yes, several aspects might impact future uses:
- The dataset is limited to Reddit submission statements, which might not fully represent all forms of online conspiracy discussions.
- The oversampling of potentially conspiracy-related content means the dataset does not reflect the natural prevalence of such content on Reddit.
- The inter-annotator agreement for conspiracy labels (α=0.58) suggests some level of subjectivity and potential noise in the classification.
- The annotators were selected from specific English-speaking countries, potentially introducing cultural biases in the identification of conspiracy theories.

### Are there tasks for which the dataset should not be used?

The dataset might not be suitable for directly assessing the overall prevalence of conspiracy theories on Reddit or other platforms due to the oversampling strategy. It is also important to be cautious when generalizing findings based on this dataset to other types of online communication beyond Reddit submission statements. Conspiracy theorizing is a stigmatized practice and participation in conspiracy theory communities provides benefits to their members (including social, psychological, and epistemic), therefore applications of the dataset should not marginalize individuals.

### Any other comments?
The dataset's focus on psychologically grounded markers offers a unique approach to understanding and detecting conspiracy theories. The topic-agnostic nature of the dataset allows for analysis across a broad range of everyday social media conversations.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

Yes, the dataset is made available through Zenodo.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

The dataset is distributed in jsonl format through the Zenodo repository and is expected to have a persistent DOI.

### When will the dataset be distributed?
2025/03/31

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset is released under CC-BY license.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Reddit's terms of service generally apply to the original source data.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?


### Any other comments?

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
The dataset is hosted on Zenodo, a platform for open research data. The creators, Mattia Samory, Felix Soldner, and Veronika Batzdorfer, are responsible for its maintenance and updates.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
Creators can be contacted via email at their current affiliation.

### Is there an erratum?

Users should refer to the Zenodo page for any updates or corrections.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Updates to the dataset are planned and will be enacted by introducing new version of the dataset on Zenodo. 

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

There are no specific retention limits of the data shared. Reddit's platform policies govern the retention of the original comments.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Zenodo maintains versions of datasets. The policy on supporting older versions will be managed through the Zenodo platform and communicated on the dataset's Zenodo page.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

The dataset is shared under a CC-BY license, encouraging reuse and adaptation. Researchers interested in extending the dataset are welcome to contact the creators.

### Any other comments?