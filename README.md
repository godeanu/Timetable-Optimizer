# Timetable-Optimizer
Using Hill-Climbing or Monte-Carlo Tree Search to generate university timetables, accommodating different constraints and preferences

# How it works
The code reads multiple soft constraints from an input yaml file.
One input file contains the following:
* Intervale - available intervals for teaching
* Materii - number of subjects that need to be taught that day, followed by the total number of students for each of those subjects
* Profesori - the names of the teachers. Each of them have their own constraints: if they prefer not teaching during a certain interval, that interval
will have a '!' before it. Same applies if they prefer not teaching during a certain day
            -  The teahcers will also have a list of the subjects they can teach
* Sali - room names. Each room has maximum student capacity and can only hold certain subjects
* Zile - available days for teaching
* Bonus - some teachers might prefer having a break smaller than a certain amount of hours between their classes

Our purpose is to minimize these soft constraints, so that each teacher can teach during their preferred hours and days, and keep their breaks how they want them.
There are obviously some hard constraints as well, which must be respected at all times: a teacher cannot teach more than 7 classes per week and cannot teach two subjects at the same time.
We must make sure that the entire number of students for each subjects will have a seat that day. We must respect the room maximum capacity and the subjects that can be taught in it.

The code should be run like this:
***python3 optimizer.py {input_file_name} {algorithm}*** , where algorithm can either be hc ( for hill-climbing), or mcts (for monte-carlo tree search)
example: ***python3 optimizer.py dummy hc***
