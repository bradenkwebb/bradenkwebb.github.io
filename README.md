# About This Website

This is a static website. The content itself is built by [Jekyll](https://jekyllrb.com/) (a static site generator) and served by [GitHub Pages](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll). It currently uses version 2.5 of the [Minima Theme](https://github.com/jekyll/minima/blob/v2.5.0/README.md).

Switching to other themes is certainly possible, but requires substantially more work than just changing a line or two of code. It may be easier, more straightforward, and more desirable to simply adjust the layout and styling files of the current theme than to attempt to transfer content to a new theme.

## Setup

This repository was initially created in a standard Windows 11 environment with a simple, out-of-the-box installation of Ruby, Bundler, and Jekyll. I did not record the exact steps I took to set up that environment, but following [this tutorial](https://jekyllrb.com/docs/step-by-step/01-setup/) should be helpful.

In particular, once you have Ruby and Bundler installed, double-check that they are working and on your path by running `where bundle` or `which bundle` in Git Bash. Assuming that returns a meaningful path to the binary, you should then run

```bash
bundle install
```

to install the gems listed in the Gemfile, including Jekyll. To confirm that the Minima theme has been properly installed alongside Jekyll, run `bundle show minima`, which should display the directory location where Minima is installed. Opening hte `README.md` within that directory should also display the Minima documentation.

To view changes to your content in real-time, run the following:

```bash
bundle exec jekyll serve
```

## See also

This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for Minima at GitHub:
[jekyll][jekyll-organization] /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at GitHub:
[jekyll][jekyll-organization] /
[jekyll](https://github.com/jekyll/jekyll)

[jekyll-organization]: https://github.com/jekyll

- [Jekyll](https://jekyllrb.com/)
- [Bundler](https://bundler.io/)
- [Minima Theme v2.5.0](https://github.com/jekyll/minima/blob/v2.5.0/README.md)
- [Markdown Style Guide](https://google.github.io/styleguide/docguide/style.html#minimum-viable-documentation)
