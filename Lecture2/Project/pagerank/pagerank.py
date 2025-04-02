import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    #    raise NotImplementedError 
    ans_dic = {}
    chil_len = len(corpus[page])
    length = len(corpus)
    
    # prob = damping_factor / chil_len
    # for html in corpus[page]:
    #     ans_dic[html] = prob
    if chil_len > 0:
        prob = damping_factor / chil_len
        for html in corpus[page]:
            ans_dic[html] = prob
    else:
        prob_no_links = damping_factor / length 
        for html in corpus:
            ans_dic[html] = ans_dic.get(html, 0) + prob_no_links
               
    prob = (1 - damping_factor) / length
    for html in corpus:
        ans_dic[html] = ans_dic.get(html, 0) + prob
        
    return ans_dic

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #    raise NotImplementedError
    page_visit = {page: 0 for page in corpus}
    all_pages = list(corpus.keys())
    curr_page = random.choice(all_pages)
    
    for _ in range(n):
        page_visit[curr_page] += 1
        probabilities = transition_model(corpus, curr_page, damping_factor)
        next_pages = list(probabilities.keys())
        weights = list(probabilities.values())
        curr_page = random.choices(next_pages, weights=weights, k=1)[0]
    
    paegranks = {}
    for page in page_visit:
        paegranks[page] = page_visit[page] / n
        
    return paegranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #   raise NotImplementedError
    # def check_convergence(new_dic, dic):
    #     for page in new_dic:
    #         if abs(new_dic[page] - dic[page]) > 0.001:
    #             return False
    #     return True
    
    # def upgrade_pr(page):
    #     new_pr = (1 - damping_factor) / n
        
    #     links = list(corpus[page])
    #     prob_sum = 0

    #     for linked_page in links:
    #         # pageranks[linked_page]  = PR(linked_page), len(corpus[linked_page]) = NumLinks(linked_page)
    #         prob_sum += damping_factor * pageranks[linked_page] / len(corpus[linked_page])
        
    #     new_pr += prob_sum
        
    #     return new_pr
            
    pageranks = {}
    
    # Init ervery page pr = 1 / n 
    n = len(corpus)
    for page in corpus:
        pageranks[page] = 1 / n
    # Upgrade PR(i) 
    incoming_links = {} 
    for page in corpus:
        incoming_links[page] = set()
    for page in corpus:
        for link in corpus[page]:
            incoming_links[link].add(page)

    while True:
        precision = True
        new_pageranks = {} 
        for page in corpus:
            new_pr = (1 - damping_factor) / n
            linking_prob_sum = 0
            for i in corpus: # Iterate through all potential linking pages (i)
                num_outlinks = len(corpus[i])
                if num_outlinks == 0:
                    # Page 'i' has no links, contributes PR(i)/N to all pages (including 'page')
                    linking_prob_sum += pageranks[i] / n
                elif page in corpus[i]:
                    # Page 'i' links to 'page', contributes PR(i)/NumLinks(i)
                    linking_prob_sum += pageranks[i] / num_outlinks
                # else: Page 'i' has links but doesn't link to 'page', contributes 0
                
            new_pr += damping_factor * linking_prob_sum
            new_pageranks[page] = new_pr # hpgrade new_pageranks 

        for page in pageranks: # checking convergence
            if abs(new_pageranks[page] - pageranks[page]) > 0.001:
                precision = False
                break
        pageranks = new_pageranks # upgrade pageranks
        if precision:
            break
    
    return pageranks
if __name__ == "__main__":
    main()

# Conclusion: I believe the most challenging part of this Project is the iterate_pagerank function, with two main pitfalls: 
# 1. Not fully understanding the requirements, which leads to not updating pageranks separately, causing convergence issues;
# 2. The boundary conditions of the convergence function were not clearly considered, 
# especially when: a page has links but there exists pages with no links in the corpus, in this case its contribution to the non-linked page is 0.