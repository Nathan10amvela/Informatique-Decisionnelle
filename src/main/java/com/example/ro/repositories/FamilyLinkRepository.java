package com.example.ro.repositories;

import com.example.ro.models.FamilyLink;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface FamilyLinkRepository  extends JpaRepository<FamilyLink, Integer> {
    List<FamilyLink> findByFamilyTreeId(int familyTreeId);
    Optional<FamilyLink> findBySourceIdAndTargetIdAndFamilyTreeId(int sourceId, int targetId, int familyTreeId);

}
