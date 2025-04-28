package com.example.ro.services;

import com.example.ro.dto.requestDTO.FamilyTreeDTO;
import com.example.ro.dto.requestDTO.UpdateFamilyTreeDTO;
import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.models.FamilyTree;
import com.example.ro.repositories.FamilyTreeRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
@AllArgsConstructor
public class FamilyTreeService {

    private final FamilyTreeRepository familyTreeRepository;



    public ApiError createFamilyTree(FamilyTreeDTO dto) {
        ApiError apiError = new ApiError();

        FamilyTree familyTree = new FamilyTree();
        familyTree.setName(dto.getName());
        familyTree.setCreator(dto.getCreator());
        familyTree.setDescription(dto.getDescription());
        familyTree.setGeographicOrigin(dto.getGeographicOrigin());
        familyTree.setCreationDate(Instant.now());
        familyTree.setLastModifiedDate(Instant.now());
        familyTree.setPrivate(dto.getIsPrivate());

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

        if (dto.getIsPrivate() != null){
            familyTree.setPrivate(dto.getIsPrivate());
        }

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
