package com.example.ro.services;

import com.example.ro.dto.requestDTO.FamilyTreeDTO;
import com.example.ro.dto.requestDTO.UpdateFamilyTreeDTO;
import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.models.FamilyTree;
import com.example.ro.repositories.FamilyTreeRepository;
import com.example.ro.repositories.PersonRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
@AllArgsConstructor
public class FamilyTreeService {

    private final FamilyTreeRepository familyTreeRepository;
    private final PersonRepository personRepository;


    public ApiError createFamilyTree(FamilyTreeDTO dto) {
        ApiError apiError = new ApiError();

        FamilyTree familyTree = new FamilyTree();
        familyTree.setName(dto.getName());
        familyTree.setDescription(dto.getDescription());
        familyTree.setGeographicOrigin(dto.getGeographicOrigin());
        familyTree.setCreationDate(Instant.now());
        familyTree.setLastModifiedDate(Instant.now());
        familyTree.setCreator(dto.getCreator());

        FamilyTree saved = familyTreeRepository.save(familyTree);

        if (saved != null) {
            apiError.setValue("200");
            apiError.setText("Family tree created successfully");
            apiError.setData(saved);
        } else {
            apiError.setValue("500");
            apiError.setText("Failed to create family tree");
        }

        return apiError;
    }

    public ApiError updateFamilyTree(int id, UpdateFamilyTreeDTO dto) {
        ApiError apiError = new ApiError();

        FamilyTree familyTree = familyTreeRepository.findById(id).orElse(null);
        if (familyTree == null) {
            apiError.setValue("404");
            apiError.setText("Family tree not found");
            return apiError;
        }

        if(dto.getName() != null) {
            familyTree.setName(dto.getName());
        }
       if (dto.getDescription() != null) {
           familyTree.setDescription(dto.getDescription());
       }
        if (dto.getGeographicOrigin() != null){
            familyTree.setGeographicOrigin(dto.getGeographicOrigin());
        }

        familyTree.setLastModifiedDate(Instant.now());


        FamilyTree updated = familyTreeRepository.save(familyTree);
        if (updated != null) {
            apiError.setValue("200");
            apiError.setText("Family tree updated successfully");
            apiError.setData(updated);
        } else {
            apiError.setValue("500");
            apiError.setText("Failed to update family tree");
        }

        return apiError;

    }

    public ApiError deleteFamilyTree(int id) {
        ApiError apiError = new ApiError();

        FamilyTree familyTree = familyTreeRepository.findById(id).orElse(null);
        if (familyTree == null) {
            apiError.setValue("404");
            apiError.setText("Family tree not found");
            return apiError;
        }

        familyTreeRepository.deleteById(id);
        apiError.setValue("204");
        apiError.setText("Family tree deleted successfully");
        return apiError;

    }
}
