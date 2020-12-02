# Understanding the Tradeoffs in Client-Side Privacy for Speech Recognition

Code for respective [paper](http://www.cs.cmu.edu/~peterw1/website_files/privacy.pdf).

## Background

Existing approaches to ensuring privacy of user speech data primarily focus on server-side approaches. While improving server-side privacy reduces certain security concerns, users still do not retain control over whether privacy is ensured on the client-side. In this paper, we define, evaluate, and explore techniques for client-side privacy in speech recognition, where the goal is to preserve privacy on raw speech data before leaving the client’s device. We first formalize several tradeoffs in ensuring client-side privacy between performance, compute requirements, and privacy. Using our tradeoff analysis, we perform a large-scale empirical study on existing approaches and find that they fall short on at least one metric. Our results call for more research in this crucial area as a step towards safer real-world deployment of speech recognition systems at scale across mobile devices

## Notes

 - We used [wav2vec 2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) as our ASR model.

 - [Link](https://github.com/jjery2243542/voice_conversion) to GAN voice conversion model.

 - [Link](https://github.com/jjery2243542/adaptive_voice_conversion) to VAE voice conversion model.

 - We used [REAPER](https://github.com/google/REAPER) and [Rubber Band](github.com/breakfastquay/rubberband) to build our signal processing approach. Details are described in our [paper](http://www.cs.cmu.edu/~peterw1/website_files/privacy.pdf).

 - Run `src/main.py` to train the adversary classifier.