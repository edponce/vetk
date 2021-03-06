# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import re
sys.path.append(os.path.abspath('..'))
import vetkit as pkg


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx='1.7'
try:
    with open('../extras_requirements.txt') as fd:
        regex = re.compile(r'^Sphinx[<>=]+(\d+[\.?\d*]*).*[\r\n]')
        for line in fd:
            match = regex.fullmatch(line)
            if match:
                needs_sphinx = match.group(1)
                break
except Exception:
    pass

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = pkg.__name__
author = pkg.__author__
copyright = pkg.__copyright__

title = "{} Documentation".format(pkg.__title__)
description = pkg.__description__

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pkg.__version__
# The full version, including alpha/beta/rc tags.
release = pkg.__version__

# The language for content autogenerated by Sphinx. Refer to documentation for
# a list of supported languages. This is also used if you do content
# translation via gettext catalogs. Usually you set "language" from the command
# line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This patterns also
# effect to html_static_path and html_extra_path
exclude_patterns = []

# The default options for autodoc directives. They are applied to all autodoc
# directives automatically. It must be a dictionary which maps option names to
# the values.
#
# 'member-order' selects if automatically documented members are sorted
# alphabetical (value 'alphabetical'), by member type (value 'groupwise') or by
# source order (value 'bysource'). The default is alphabetical.
autodoc_member_order = 'bysource'

# Special members are any methods or attributes that start with and end with a
# double underscore. Any special member with a docstring will be included in the
# output, if 'napoleon_include_special_with_doc' is set to True.
napoleon_include_special_with_doc = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for a
# list of builtin themes.
#
# html_theme = 'basic'        # unstyled layout
# html_theme = 'traditional'  # old Python documentation
# html_theme = 'classic'      # Python2-like documentation
# html_theme = 'sphinxdoc'    # Sphinx documentation
# html_theme = 'alabaster'    # modified "Kr" Sphinx theme
# html_theme = 'nature'       # greenish theme
# html_theme = 'pyramid'      # Pyramid web framework theme
# html_theme = 'bizstyle'     # simple bluish theme
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
if html_theme in ('basic', 'traditional', 'sphinxdoc', 'nature', 'pyramid'):
    html_theme_options = {
        'nosidebar': 'false',
        'sidebarwidth': '200',  # in pixels
    }
elif html_theme == 'classic':
    html_theme_options = {
        'rightsidebar': 'false',
        'stickysidebar': 'false',
        'externalrefs': 'false',
    }
elif html_theme == 'bizstyle':
    html_theme_options = {
        'nosidebar': 'false',
        'rightsidebar': 'false',
        'sidebarwidth': '200',  # in pixels
    }
else:
    html_theme_options = {}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names to
# template names.
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # Remove extra blank pages
    'extraclassoptions': 'openany, oneside',

    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples (source start
# file, target name, title, author, documentclass [howto, manual, or own
# class]).
latex_documents = [
    (master_doc, project + '.tex', title, author, 'manual')
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples (source start file, name,
# description, authors, manual section).
man_pages = [
    (master_doc, project, title, [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples (source start
# file, target name, title, author, dir menu entry, description, category)
texinfo_documents = [
    (master_doc, project, title, author, project, description, 'Miscellaneous')
]
