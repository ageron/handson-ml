# Guide to contributing

This book is developed collaboratively and openly, here on GitHub. We accept comments, contributions and corrections from all.

## Current Project STATUS
**CONTENT FREEZE - FIRST EDITION IN PRODUCTION**

## Contributing with a Pull Request

Before contributing with a Pull Request, please read the current **PROJECT STATUS**.

If the current **PROJECT STATUS** is **CONTENT FREEZE**, please keep these points in mind;

* Please submit only PRs for errors that a non-domain-expert copy editor might miss. Do not submit PRs for typos, grammar and syntax, as those are part of the copy editors job.
* Please don't merge code. Any changes will have to be applied manually (by the Author) after copy edit and before final proof, if the copy editor doesn't catch the same errors.


## Chat with the authors

You can chat with the authors and editors on [COMPLETE...]

## License and attribution

All contributions must be properly licensed and attributed. If you are contributing your own original work, then you are offering it under a CC-BY license (Creative Commons Attribution). You are responsible for adding your own name or pseudonym in the Acknowledgments section in the [Preface](preface.asciidoc), as attribution for your contribution.

If you are sourcing a contribution from somewhere else, it must carry a compatible license. The book will initially be released under a CC-BY-NC-ND license which means that contributions must be licensed under open licenses such as MIT, CC0, CC-BY, etc. You need to indicate the original source and original license, by including an asciidoc markup comment above your contribution, like this:

```asciidoc
////
Source: https:// [...]
License: CC0 [...]
Added by: @[...]
////
```

The best way to contribute to this book is by making a pull request:

1. Login with your GitHub account or create one now
2. [Fork] the repository. Work on your fork.
3. Create a new branch on which to make your change, e.g. `git checkout -b my_code_contribution`, or make the change on the `develop` branch.
4. Please do one pull request PER file, to avoid large merges. Edit the file where you want to make a change or create a new file in the `contrib` directory if you're not sure where your contribution might fit.
5. Edit `preface.asciidoc` and add your own name to the list of contributors under the Acknowledgment section. Use your name, or a GitHub username, or a pseudonym.
6. Commit your change. Include a commit message describing the correction.
7. Submit a pull request against the handon-ml repository.

## Contributing with an issue

If you find a mistake and you're not sure how to fix it, or you don't know how to do a pull request, then you can file an Issue. Filing an Issue will help us see the problem and fix it.

Create a [new Issue](https://github.com/ageron/handson-ml/issues/new) now!

## Heading styles normalization across the book
[COMPLETE]

## Thanks

We are very grateful for the support of the entire community. With your help, this will be a great book that can help thousands of developers get started and eventually. Thank you!
