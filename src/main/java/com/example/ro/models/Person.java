package com.example.ro.models;

import com.example.ro.enumeration.Gender;
import com.example.ro.enumeration.Role;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


import java.time.Instant;
import java.util.List;

@Data
@Entity
@AllArgsConstructor
@NoArgsConstructor
public class Person {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private String lastName;

    private String firstName;

    //private Instant birthDate;

    private String birthPlace;

    @Enumerated(EnumType.STRING)
    private Gender gender;

    @Enumerated(EnumType.STRING)
    private Role role;

    @ManyToOne
    @JoinColumn(name = "tree_id", nullable = false)
    private FamilyTree familyTree;

    @OneToMany(mappedBy = "source")
    private List<FamilyLink> outgoingLinks;

    @OneToMany(mappedBy = "target")
    private List<FamilyLink> incomingLinks;
}
