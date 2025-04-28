package com.example.ro.controllers;


import com.example.ro.dto.requestDTO.FamilyTreeDTO;
import com.example.ro.dto.requestDTO.UpdateFamilyTreeDTO;
import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.services.FamilyTreeService;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/family-trees")
@AllArgsConstructor
public class FamilyTreeController {


    private final FamilyTreeService familyTreeService;

    @PostMapping
    public ApiError createFamilyTree(@RequestBody FamilyTreeDTO dto) {
        return familyTreeService.createFamilyTree(dto);
    }

    @PatchMapping("/{id}")
    public ApiError updateFamilyTree(@PathVariable int id, @RequestBody UpdateFamilyTreeDTO dto) {
        return familyTreeService.updateFamilyTree(id, dto);
    }

    @DeleteMapping("/{id}")
    public ApiError deleteFamilyTree(@PathVariable int id) {
        return familyTreeService.deleteFamilyTree(id);
    }
}

