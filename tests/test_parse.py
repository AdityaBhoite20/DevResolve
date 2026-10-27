# test_parse.py - checks that the HTML parser keeps the text, headings and code, and removes page noise
from parse_docs import html_to_sections

SAMPLE = """<html><head><title>3.1. Cross-validation &#8212; scikit-learn 1.8.0 documentation</title></head>
<body><nav>menu</nav><article class="bd-article"><section id="cv">
<h1>3.1. Cross-validation<a class="headerlink" href="#cv">#</a></h1>
<p>Use <code>cross_val_score</code> to evaluate a model.</p>
<div class="highlight-python"><div class="highlight"><pre>score = 1
print(score)
</pre></div></div>
<section id="sub"><h2>Sub section</h2><ul><li>first item</li><li>second item</li></ul></section>
<footer class="bd-footer-article">FOOTER NOISE</footer>
</section></article></body></html>"""


def test_title_version_and_sections():
    title, version, sections = html_to_sections(SAMPLE)
    assert title == "3.1. Cross-validation"  # the ending is removed
    assert version == "1.8.0"  # version found in the title
    assert len(sections) == 2  # one section per heading
    assert sections[0]["anchor"] == "cv"  # anchor id kept for links
    assert sections[1]["heading"] == "Sub section"


def test_noise_removed_and_code_kept():
    title, version, sections = html_to_sections(SAMPLE)
    text = str(sections)
    assert "FOOTER NOISE" not in text  # footer removed
    assert any("cross_val_score" in block for block in sections[0]["blocks"])  # normal text kept
    assert any("score = 1\nprint(score)" in block for block in sections[0]["blocks"])  # code keeps its line breaks
