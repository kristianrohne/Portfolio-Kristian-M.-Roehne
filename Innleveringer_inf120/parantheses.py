# -*- coding: utf-8 -*-
"""
__author__ = Kristian Mathias Røhne
__email__ = kristian.mathias.rohne@nmbu.no
"""
import numpy as np


def check_parantheses(text):
    
    parantheses=["(",")"]
    l=[]
    d={}
    
    for w in text:
        if w in parantheses:
            l.append(w)
            
    for i in range(len(l)):
        if l[i] == '(':
            l[i] = 1
        elif l[i] == ')':
            l[i] = -1
            
    l2= np.cumsum(l)
    
    d["max depth"]= max(l2)
    
    if len(l2)%2==0:
        d["is valid"]=True
    else:
        d["is valid"]=False
        
    if sum(l)==0:
        d["is balanced"]=True
    else:
        d["is balanced"]=False
        
    return d



lisp = '''
(defun LaTeX-newline ()
  "Start a new line potentially staying within comments.
This depends on `LaTeX-insert-into-comments'."
  (interactive)
  (if LaTeX-insert-into-comments
      (cond ((and (save-excursion (skip-chars-backward " \t") (bolp))
                  (save-excursion
                    (skip-chars-forward " \t")
                    (looking-at (concat TeX-comment-start-regexp "+"))))
             (beginning-of-line)
             (insert (buffer-substring-no-properties
                      (line-beginning-position) (match-end 0)))
             (newline))
            ((and (not (bolp))
                  (save-excursion
                    (skip-chars-forward " \t") (not (TeX-escaped-p)))
                  (looking-at
                   (concat "[ \t]*" TeX-comment-start-regexp "+[ \t]*")))
             (delete-region (match-beginning 0) (match-end 0))
             (indent-new-comment-line))
            ;; `indent-new-comment-line' does nothing when
            ;; `comment-auto-fill-only-comments' is non-nil, so we
            ;; must be sure to be in a comment before calling it.  In
            ;; any other case `newline' is used.
            ((TeX-in-comment)
             (indent-new-comment-line))
            (t
             (newline)))
    (newline)))
'''

res = check_parantheses(lisp)
print(res)

"""
max_dept: 8
is_valid: True
is_balanced: True
"""     
    
    
            