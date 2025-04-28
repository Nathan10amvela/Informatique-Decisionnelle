package com.example.ro.models;



import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Entity
@AllArgsConstructor
@NoArgsConstructor
public class FamilyLink {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private int weight;
    private String relationType;

    @ManyToOne
    @JoinColumn(name = "source_id", nullable = false)
    private Person source;

    @ManyToOne
    @JoinColumn(name = "target_id", nullable = false)
    private Person target;

    @ManyToOne
    @JoinColumn(name = "tree_id", nullable = false)
    private FamilyTree familyTree;
}

