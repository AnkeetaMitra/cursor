% ===================================================
% EXAM TIMETABLE PLANNER USING PLANNING GRAPH (Prolog)
% ===================================================

% -----------------------------
% STEP 1: Basic Room and Exam Facts (User-Defined)
% -----------------------------

:- dynamic room/4.
:- dynamic room_computers/2.
:- dynamic exam/4.
:- dynamic assigned/4.
:- dynamic s_state/2.
:- dynamic a_action/2.
:- dynamic persistence/2.

% room(RoomID, TotalCapacity, IsTiered (yes/no), SeatsPerRow).
add_room(RoomID, Capacity, Tiered, SeatsPerRow) :-
    assertz(room(RoomID, Capacity, Tiered, SeatsPerRow)).

% room_computers(RoomID, NumComputers).
add_room_computers(RoomID, NumComputers) :-
    assertz(room_computers(RoomID, NumComputers)).

% exam(ExamID, NumStudents, DurationHours, Type).
add_exam(ExamID, NumStudents, DurationHours, Type) :-
    assertz(exam(ExamID, NumStudents, DurationHours, Type)).

% -----------------------------
% STEP 2: Effective Capacity Logic
% -----------------------------

effective_capacity(RoomID, offline, EffectiveCapacity) :-
    room(RoomID, Capacity, IsTiered, SeatsPerRow),
    Rows is Capacity // SeatsPerRow,
    (
        IsTiered == yes ->
            UsableRows is (Rows + 1) // 2, % Leave one row in between
            UsableSeatsPerRow is SeatsPerRow // 2  % Leave one seat gap on both sides
        ;
            UsableRows is Rows,
            UsableSeatsPerRow is SeatsPerRow // 2
    ),
    EffectiveCapacity is UsableRows * UsableSeatsPerRow.

effective_capacity(RoomID, online, EffectiveCapacity) :-
    room_computers(RoomID, Capacity),
    EffectiveCapacity is Capacity.

% -----------------------------
% STEP 3: Assignment Logic with Time Slots
% -----------------------------

room_available(RoomID, TimeSlot) :-
    \+ assigned(RoomID, TimeSlot, _, _).

room_can_host(RoomID, ExamID) :-
    exam(ExamID, Students, _, Type),
    effective_capacity(RoomID, Type, Cap),
    Cap >= Students.

exam_not_assigned(ExamID) :-
    \+ assigned(_, _, ExamID, _).

assign_exam(ExamID, RoomID, TimeSlot, Level) :-
    exam_not_assigned(ExamID),
    room_available(RoomID, TimeSlot),
    room_can_host(RoomID, ExamID),
    exam(ExamID, _, Duration, _),
    assertz(assigned(RoomID, TimeSlot, ExamID, Duration)),
    format(atom(Act), 'assign(~w,~w,~w)', [ExamID, RoomID, TimeSlot]),
    assertz(a_action(Level, Act)),
    format(atom(Assigned), 'assigned(~w,~w,~w,~w)', [RoomID, TimeSlot, ExamID, Duration]),
    format(atom(Neg1), 'not(exam_not_assigned(~w))', [ExamID]),
    format(atom(Neg2), 'not(room_available(~w,~w))', [RoomID, TimeSlot]),
    assertz(s_state(Level+1, Assigned)),
    assertz(s_state(Level+1, Neg1)),
    assertz(s_state(Level+1, Neg2)).

% -----------------------------
% STEP 4: Sequential Plan Execution
% -----------------------------

assign_all_exams([], _, _).
assign_all_exams([Exam|Rest], TimeSlots, Level) :-
    member(TimeSlot, TimeSlots),
    room(RoomID, _, _, _),
    assign_exam(Exam, RoomID, TimeSlot, Level),
    NextLevel is Level + 1,
    propagate_persistence(Level, NextLevel),
    assign_all_exams(Rest, TimeSlots, NextLevel).

run_timetable_plan(TimeSlots) :-
    findall(E, exam(E, _, _, _), Exams),
    initialize_state(0),
    assign_all_exams(Exams, TimeSlots, 0).

% -----------------------------
% STEP 5: Action Schema Representation
% -----------------------------

action_schema(assign(Exam, Room, Time),
    [exam_not_assigned(Exam), room_available(Room, Time), room_can_host(Room, Exam)],
    [assigned(Room, Time, Exam, Duration), not(exam_not_assigned(Exam)), not(room_available(Room, Time))]) :-
        exam(Exam, _, Duration, _).

% -----------------------------
% STEP 6: Goal Checking
% -----------------------------

goal_state :-
    findall(E, exam(E, _, _, _), Exams),
    forall(member(X, Exams), \+ exam_not_assigned(X)).

% -----------------------------
% STEP 7: State Initialization & Persistence
% -----------------------------

initialize_state(Level) :-
    findall(Fact, exam_not_assigned(Fact), NotAssignedList),
    forall(member(E, NotAssignedList), (
        format(atom(Lit), 'exam_not_assigned(~w)', [E]),
        assertz(s_state(Level, Lit)))).

propagate_persistence(Current, Next) :-
    findall(Lit, s_state(Current, Lit), CurrentLiterals),
    forall(member(L, CurrentLiterals), (
        \+ s_state(Next, L) ->
            assertz(s_state(Next, L)),
            format(atom(P), 'persist(~w)', [L]),
            assertz(persistence(Next, P))
        ; true
    )).

% -----------------------------
% STEP 8: Query Utilities for Planning Graph
% -----------------------------

print_state(Level) :-
    format('State S~w:\n', [Level]),
    forall(s_state(Level, L), (write('- '), writeln(L))).

print_action(Level) :-
    format('Action A~w:\n', [Level]),
    forall(a_action(Level, A), (write('* '), writeln(A))).

print_persistence(Level) :-
    format('Persistence at S~w:\n', [Level]),
    forall(persistence(Level, P), (write('+ '), writeln(P))).
