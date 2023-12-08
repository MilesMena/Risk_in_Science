import preferential_attachment_sim as pas

authors = ['Einstein','Curie','Newton','Lovelace','Darwin','Bill Nye']

papers = pas.init_web(authors)

sim = pas.take_steps_multiple_cite(papers,authors, 60, max_cite = 3)

print(sim)