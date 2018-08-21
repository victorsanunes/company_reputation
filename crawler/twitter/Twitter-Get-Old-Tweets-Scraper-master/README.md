# Old Tweets Scraper Using Python

---

[Twitter's API](https://dev.twitter.com/rest/reference/get/search/tweets) no longer allows the retrieval of tweets past 7 days old using apps. Some third party apps are available to allow retrieval of tweets past 7 days old but you need to spend money in most cases. This script was written to enable the retrieval of tweets past the limitation of the API. Most of the code were written and modified based on @Jefferson-Henrique's [scraper](https://github.com/Jefferson-Henrique/GetOldTweets-python). The code was modified to support Python 3 and to comply to pep coding standards.

---

#### Installation
Download the source code using git. Install python dependencies by typing `make install` in the terminal.

---

#### Command Line Usage
The script can be used by typing `python main.py` and requires an argument passed with it. For more information you may type `python main.py -help` in the command line.

---

#### Parameters
The script can be run with 5 arguments. The **query** and/or **username** argument must be used. The other arguments are optional.

| Argument | Format | Usage Example |
|---|---|---|
| query | |`python main.py --query "#MarcosNotAHero" --max-tweets 1`|
| username| |`python main.py --username "barackobama" --max-tweets 1`|
|since|YYYY-MM-DD|`python main.py --query "Benham Rise" --since 2016-01-01 --max-tweets 1`|
|since|YYYY-MM-DD|`python main.py --query "Benham Rise" --until 2017-04-01 --max-tweets 1`|
|max-tweets (*default*: 100)| |`python main.py --query "#MarcosNotAHero" --max-tweets 1`|


---
#### Code Usage
You may also write your own code based on the controllers and models inside the [scraper](https://github.com/PJHRobles/Twitter-Old-Tweets-Scraper/tree/master/scraper) folder.
